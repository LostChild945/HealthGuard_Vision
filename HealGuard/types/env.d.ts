declare global {
  namespace NodeJS {
    interface ProcessEnv {
      EXPO_PUBLIC_API_KEY: string;
      EXPO_PUBLIC_API_BASE_URL: string;
    }
  }
}

export {};
