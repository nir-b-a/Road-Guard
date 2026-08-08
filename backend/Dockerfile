# Road Guard - Node.js/Express REST API
#
# node:22 (LTS), not 20: AWS SDK v3 already warns that it will soon require >=22,
# and that SDK is what mints every R2 presigned URL.
# alpine keeps the image small (the backend never touches video bytes - those go
# straight from the client to Cloudflare R2 via presigned URLs).
FROM node:22-alpine

WORKDIR /app

# Copy manifests first so `npm ci` is cached and only re-runs when deps change.
COPY package.json package-lock.json ./

# `npm ci` installs exactly the lockfile (reproducible); --omit=dev skips jest,
# supertest, nodemon and mongodb-memory-server, which the container never runs.
RUN npm ci --omit=dev

COPY . .

# Do not bake secrets into the image: .env is supplied at run time by compose.
EXPOSE 5000

# server.js directly, NOT `npm start` - that script rewrites .env with a
# WSL-specific host IP, which is wrong (and destructive) inside a container.
CMD ["node", "server.js"]
