# Road Guard - authority web dashboard (React 19 + Vite 8 + Tailwind)
#
# Two stages: the build stage compiles the SPA, the run stage serves the static
# bundle. Building in Docker also proves the production build works - a demo that
# only ever ran `npm run dev` can still fail at `vite build`.
FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
# Dev deps ARE needed here: vite, tailwind and postcss all run at build time.
RUN npm ci

COPY . .

# VITE_API_BASE is baked in at build time (Vite inlines import.meta.env). Defaults
# to localhost:5000 so the browser on the host talks to the published backend port.
ARG VITE_API_BASE=http://localhost:5000/api
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build


FROM node:22-alpine AS run
WORKDIR /app

# `vite preview` serves dist/ over plain HTTP - enough for a demo and it keeps the
# stack to one toolchain (no nginx config to get wrong the morning of the meeting).
COPY --from=build /app/dist ./dist
COPY --from=build /app/package.json /app/package-lock.json ./
COPY --from=build /app/vite.config.js ./
RUN npm ci

EXPOSE 4173
# --host binds 0.0.0.0 so the port is reachable from outside the container.
CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "4173"]
