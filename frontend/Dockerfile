# Use the official Node.js 20 image as the base image
FROM node:20

# Set the working directory
WORKDIR /app

# Copy the package.json and package-lock.json (or yarn.lock) to the working directory
COPY frontend/package*.json ./

# Install the project dependencies
RUN npm install

# Copy the rest of the application files to the working directory
COPY frontend/ ./

# Build the application
RUN npm run build

# Install a simple web server to serve the built files
RUN npm install -g serve

# Set the command to serve the built files
CMD ["serve", "-s", "dist"]

# Expose port 5000 for the web server
EXPOSE 5000
# Use the official Node.js 20 image as the base image
FROM node:20

# Set the working directory
WORKDIR /app

# Copy the package.json and package-lock.json (or yarn.lock) to the working directory
COPY frontend/package*.json ./

# Install the project dependencies
RUN npm install

# Copy the rest of the application files to the working directory
COPY frontend/ ./

# Build the application
RUN npm run build

# Install a simple web server to serve the built files
RUN npm install -g serve

# Set the command to serve the built files
CMD ["serve", "-s", "dist"]

# Expose port 5000 for the web server
EXPOSE 5000