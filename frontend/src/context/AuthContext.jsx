"use client";

import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import * as api from "@/lib/api";

export const DEFAULT_DEMO_PROFILES = [
  {
    email: "engineer@advertest.ai",
    password: "EngineerPassword123!",
    display_name: "Alex (ML Engineer)",
    avatar_url: "https://api.dicebear.com/7.x/bottts/svg?seed=alex",
    role: "ENGINEER",
  },
  {
    email: "admin@advertest.ai",
    password: "AdminPassword123!",
    display_name: "System Administrator",
    avatar_url: "https://api.dicebear.com/7.x/bottts/svg?seed=admin",
    role: "ADMIN",
  },
];

const DEFAULT_AUTH_FALLBACK = {
  user: null,
  isAuthenticated: false,
  role: "ENGINEER",
  isLoading: false,
  googleConfig: { client_id: "", configured: false, demo_profiles: DEFAULT_DEMO_PROFILES },
  isAuthModalOpen: false,
  openAuthModal: () => {},
  closeAuthModal: () => {},
  loginWithGoogle: async () => {},
  loginWithDemoProfile: async () => {},
  loginWithCredentials: async () => {},
  registerWithCredentials: async () => {},
  logout: () => {},
  switchRole: () => {},
};

const AuthContext = createContext(DEFAULT_AUTH_FALLBACK);

export function AuthProvider({ children }) {
  // Bypassing auth completely per user request
  const user = {
    email: "local@advertest.ai",
    display_name: "Local User",
    role: "ADMIN",
    avatar_url: "https://api.dicebear.com/7.x/bottts/svg?seed=local",
  };
  
  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: true,
        role: "ADMIN",
        isLoading: false,
        googleConfig: { client_id: "", configured: false, demo_profiles: DEFAULT_DEMO_PROFILES },
        isAuthModalOpen: false,
        openAuthModal: () => {},
        closeAuthModal: () => {},
        loginWithGoogle: async () => {},
        loginWithDemoProfile: async () => {},
        loginWithCredentials: async () => {},
        registerWithCredentials: async () => {},
        logout: () => {},
        switchRole: () => {},
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  return context || DEFAULT_AUTH_FALLBACK;
}
