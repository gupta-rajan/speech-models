# Stage 1: Build the application
FROM node:20.0-alpine as builder

# Install pnpm globally
RUN npm install -g pnpm

# Set working directory
WORKDIR /app

# Copy package.json and pnpm-lock.yaml to leverage Docker cache
COPY package.json pnpm-lock.yaml ./

# Install dependencies
RUN pnpm install

# Copy the rest of the application code
COPY . .

# Build the application
RUN npm run build

# Stage 2: Serve the application
FROM nginxinc/nginx-unprivileged:stable-alpine-slim

# Copy built files from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Expose port
EXPOSE 8000

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]