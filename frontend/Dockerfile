# Use the official Node image from the Docker Hub
FROM node:14

# Set working directory
WORKDIR /app

# Install dependencies
COPY package.json /app/
RUN npm install

# Copy project
COPY . /app/

# Build the application
RUN npm run build

# Run the application
CMD ["npm", "start"]