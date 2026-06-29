/**
 * r2.js — thin wrapper around Cloudflare R2 (S3-compatible) for the direct-upload
 * architecture. The big video files never flow through this server: the app PUTs them
 * straight to R2 with a presigned URL we mint here, and the GPU worker pulls them back.
 *
 * Everything degrades gracefully when R2 is NOT configured (no R2_* env vars): presign
 * helpers return a local placeholder URL and head/usage helpers act as if empty. That
 * keeps the automated tests and local-disk development working without a real bucket.
 *
 * Env vars (see .env.example):
 *   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_ENDPOINT
 *   R2_PRESIGN_EXPIRY (seconds, default 900)
 */
const {
    S3Client, HeadObjectCommand, GetObjectCommand, PutObjectCommand,
    DeleteObjectCommand, DeleteObjectsCommand, ListObjectsV2Command,
    PutBucketCorsCommand
} = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');

const BUCKET = process.env.R2_BUCKET;
const ENDPOINT = process.env.R2_ENDPOINT;
const PRESIGN_EXPIRY = parseInt(process.env.R2_PRESIGN_EXPIRY || '900', 10);

const configured = !!(BUCKET && ENDPOINT && process.env.R2_ACCESS_KEY_ID && process.env.R2_SECRET_ACCESS_KEY);

const client = configured ? new S3Client({
    region: 'auto',
    endpoint: ENDPOINT,
    credentials: {
        accessKeyId: process.env.R2_ACCESS_KEY_ID,
        secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
    }
}) : null;

const isConfigured = () => configured;

/**
 * Presigned PUT URL the app uses to upload one object directly to R2.
 * We deliberately sign a BARE PutObject (no Content-Type / Content-Length in the
 * signature): binding those would force every client (OkHttp on Android, boto3, curl)
 * to send byte-exact matching headers or get SignatureDoesNotMatch. The video size is
 * instead enforced authoritatively at /upload/complete via a HEAD on the real object.
 */
const presignPut = async (key) => {
    if (!configured) return 'local-placeholder://' + key;   // dev/test fallback
    const cmd = new PutObjectCommand({ Bucket: BUCKET, Key: key });
    return getSignedUrl(client, cmd, { expiresIn: PRESIGN_EXPIRY });
};

/** Short-lived presigned GET URL — used to let the authority dashboard stream a clip
 *  straight from R2 (the bucket stays private). */
const presignGet = async (key, { expiresIn = PRESIGN_EXPIRY } = {}) => {
    if (!configured) return null;
    const cmd = new GetObjectCommand({ Bucket: BUCKET, Key: key });
    return getSignedUrl(client, cmd, { expiresIn });
};

/**
 * Stream an object from R2 directly to an HTTP response, forwarding a Range header when present.
 * Returns { body, contentType, contentLength, contentRange, statusCode } on success, or null when
 * the object is missing. Throws on unexpected R2 errors.
 * The caller must pipe body (a Node.js Readable) to the response before it goes idle.
 */
const getObjectStream = async (key, rangeHeader) => {
    if (!configured) return null;
    const params = { Bucket: BUCKET, Key: key };
    if (rangeHeader) params.Range = rangeHeader;
    try {
        const out = await client.send(new GetObjectCommand(params));
        return {
            body:          out.Body,
            contentType:   out.ContentType || 'video/mp4',
            contentLength: out.ContentLength,
            contentRange:  out.ContentRange || null,
            statusCode:    rangeHeader ? 206 : 200,
        };
    } catch (err) {
        const code = err.$metadata && err.$metadata.httpStatusCode;
        if (code === 404 || err.name === 'NotFound' || err.name === 'NoSuchKey') return null;
        throw err;
    }
};

/** HEAD an object: returns { size } if it exists, or null if it doesn't / R2 is off. */
const headObject = async (key) => {
    if (!configured) return null;
    try {
        const out = await client.send(new HeadObjectCommand({ Bucket: BUCKET, Key: key }));
        return { size: out.ContentLength };
    } catch (err) {
        if (err.$metadata && (err.$metadata.httpStatusCode === 404 || err.name === 'NotFound')) return null;
        throw err;
    }
};

const deleteObject = async (key) => {
    if (!configured) return;
    await client.send(new DeleteObjectCommand({ Bucket: BUCKET, Key: key }));
};

/** Delete every object under a prefix (used to clean up a rejected session and to
 *  drop a session's raw files once it is processed). Best-effort: cleanup failures
 *  (e.g. token without list/delete permission) are logged, not thrown. */
const deletePrefix = async (prefix) => {
    if (!configured) return;
    try {
        let token;
        do {
            const list = await client.send(new ListObjectsV2Command({ Bucket: BUCKET, Prefix: prefix, ContinuationToken: token }));
            const objs = (list.Contents || []).map(o => ({ Key: o.Key }));
            if (objs.length) await client.send(new DeleteObjectsCommand({ Bucket: BUCKET, Delete: { Objects: objs } }));
            token = list.IsTruncated ? list.NextContinuationToken : undefined;
        } while (token);
    } catch (err) {
        console.warn('[r2] deletePrefix(' + prefix + ') failed: ' + err.message);
    }
};

// Total bucket usage, cached briefly so the /upload/init admission guard doesn't list
// the whole bucket on every request. Best-effort: if listing isn't permitted we fail
// OPEN (return 0) so uploads aren't blocked by a guard we can't evaluate.
let _usageCache = { bytes: 0, at: 0 };
const USAGE_TTL_MS = 30 * 1000;
const bucketUsageBytes = async () => {
    if (!configured) return 0;
    if (Date.now() - _usageCache.at < USAGE_TTL_MS) return _usageCache.bytes;
    try {
        let token, total = 0;
        do {
            const list = await client.send(new ListObjectsV2Command({ Bucket: BUCKET, ContinuationToken: token }));
            for (const o of (list.Contents || [])) total += (o.Size || 0);
            token = list.IsTruncated ? list.NextContinuationToken : undefined;
        } while (token);
        _usageCache = { bytes: total, at: Date.now() };
        return total;
    } catch (err) {
        console.warn('[r2] bucketUsageBytes failed (' + err.name + ') — skipping storage quota guard. '
            + 'Grant the R2 token list permission to enable it.');
        return 0;
    }
};

/** Configure CORS on the R2 bucket so browsers can fetch presigned URLs directly.
 *  Called once at server startup. Fire-and-forget — a failure is logged but never fatal. */
const configureCors = async () => {
    if (!configured) return;
    try {
        await client.send(new PutBucketCorsCommand({
            Bucket: BUCKET,
            CORSConfiguration: {
                CORSRules: [{
                    AllowedHeaders: ['*'],
                    AllowedMethods: ['GET', 'HEAD'],
                    AllowedOrigins: ['*'],
                    ExposeHeaders: ['Content-Length', 'Content-Type', 'ETag'],
                    MaxAgeSeconds: 3000,
                }]
            }
        }));
        console.log('[r2] CORS configured (GET/HEAD from any origin)');
    } catch (e) {
        console.warn('[r2] CORS setup failed:', e.message);
    }
};

module.exports = {
    isConfigured, presignPut, presignGet, headObject, getObjectStream,
    deleteObject, deletePrefix, bucketUsageBytes, configureCors
};
