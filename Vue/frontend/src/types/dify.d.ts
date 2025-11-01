declare interface Window {
  difyChatbotConfig?: {
    token: string;
    baseUrl: string;
    systemVariables?: Record<string, any>;
    userVariables?: Record<string, any>;
  }
}

export interface DifyChatbotConfig {
  token: string;
  baseUrl: string;
  systemVariables?: Record<string, any>;
  userVariables?: Record<string, any>;
} 