"use client";

import { useCallback, useEffect, useState } from "react";

const KEY = "advertest.closedloop.v1";

export const EMPTY_STORE = {
  v: 1,
  db: [], // retrain database: adversarial variants waiting for / used in training
  memory: {}, // RL memory: insight tag → recipe signature → statistics
  models: {}, // modelId → { version, fixes, baseDelta, history }
  cycles: [], // finished retrain cycles
};

function load() {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return EMPTY_STORE;
    const parsed = JSON.parse(raw);
    return parsed?.v === 1 ? { ...EMPTY_STORE, ...parsed } : EMPTY_STORE;
  } catch {
    return EMPTY_STORE;
  }
}

function persist(store) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(store));
  } catch {
    // Storage quota / private mode — the session still works in memory.
  }
}

export function useLoopStore() {
  const [store, setStore] = useState(EMPTY_STORE);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setStore(load());
    setReady(true);
  }, []);

  const update = useCallback((fn) => {
    setStore((prev) => {
      const next = fn(prev);
      persist(next);
      return next;
    });
  }, []);

  const reset = useCallback(() => {
    persist(EMPTY_STORE);
    setStore(EMPTY_STORE);
  }, []);

  return { store, update, reset, ready };
}
