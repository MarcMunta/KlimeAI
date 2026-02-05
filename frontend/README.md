# Vortex UI (Frontend)

This is the interface for the Vortex AI System. It is built with **React**, **TypeScript**, and **Vite**.

## Configuration

The frontend is configured to communicate with the local Vortex Backend.

- **API URL**: `http://localhost:8000/chat`
- **Service file**: `services/vortexService.ts`

## Commands

### Install Dependencies
```bash
npm install
```

### Run Development Server
```bash
npm run dev
```

### Build for Production
```bash
npm run build
```

## Structure

- **`services/vortexService.ts`**: Handles the streaming connection to the Python backend.
- **`components/`**: UI components for the chat interface.
- **`types.ts`**: TypeScript definitions for messages and API responses.
