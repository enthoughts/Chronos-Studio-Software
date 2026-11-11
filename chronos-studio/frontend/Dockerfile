# Chronos Studio frontend — build with Node, serve statically behind nginx.
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
LABEL org.opencontainers.image.title="Chronos Studio frontend" \
      org.opencontainers.image.description="Static SPA + /api reverse proxy"
