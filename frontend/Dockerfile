FROM node:24.15.0-slim
WORKDIR /frontend
COPY frontend/package*.json .
RUN npm ci
COPY frontend/ .
EXPOSE 3000
CMD ["npm", "run", "dev"]