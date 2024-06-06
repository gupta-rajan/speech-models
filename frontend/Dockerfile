# Stage 1: Build the application
FROM node:20 AS builder

# Set the working directory
WORKDIR /app

# Copy the package.json and package-lock.json (or yarn.lock)
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy the application source code
COPY frontend/ ./

# Build the application
RUN npm run build

# Stage 2: Serve the application with an HTTP server
FROM node:20 AS server

# Install 'serve' globally
RUN npm install -g serve

# Copy the built application from the previous stage
COPY --from=builder /app/build /app/build

# Set the working directory
WORKDIR /app

# Expose port 3000
EXPOSE 3000

# Serve the application
CMD ["serve", "-s", "build", "-l", "3000"]
