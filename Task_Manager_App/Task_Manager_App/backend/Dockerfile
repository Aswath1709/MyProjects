# Stage 1: Build
FROM node:18-alpine AS builder

WORKDIR /app

# Copy Nx monorepo root files
COPY package*.json ./
COPY nx.json ./
COPY tsconfig.base.json ./

# Copy apps and libs
COPY backend ./backend


RUN npm install

# Build only the backend app (named "backend" in Nx)
RUN npx nx build backend

# Stage 2: Run
FROM node:18-alpine

WORKDIR /app

# Copy node_modules and dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist/backend ./dist

ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

CMD ["node", "dist/main.js"]
