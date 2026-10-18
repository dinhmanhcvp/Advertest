/** @type {import('next').NextConfig} */

// The browser always calls same-origin `/api/*`; this server proxies it to the
// FastAPI backend. In DEMO_MODE the backend serves pre-computed artifacts, and
// the UI additionally falls back to its local store if the backend is offline.
const API_PROXY_TARGET = (process.env.API_PROXY_TARGET || "http://localhost:8000").replace(/\/$/, "");

const nextConfig = {
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_PROXY_TARGET}/api/:path*` }];
  },
};

export default nextConfig;
