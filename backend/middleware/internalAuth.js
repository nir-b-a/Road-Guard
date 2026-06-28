/**
 * Shared-secret guard for the /api/internal/* routes the GPU worker calls.
 * Secure-by-config: only enforced when INTERNAL_TOKEN is set in the environment, so
 * local dev and the test suite (which set no token) keep working. In production set
 * INTERNAL_TOKEN on both the server and the worker; the worker sends it as x-internal-token.
 */
const internalAuth = (req, res, next) => {
    const expected = process.env.INTERNAL_TOKEN;
    if (!expected) return next();   // not configured -> open (dev/test)
    if (req.headers['x-internal-token'] === expected) return next();
    return res.status(401).json({ success: false, message: 'Invalid internal token', data: null });
};

module.exports = { internalAuth };
