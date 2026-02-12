import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios';
import { z } from 'zod';

export class AppError extends Error {
  constructor(
    public code: string,
    message: string,
    public statusCode: number = 500,
    public originalError?: Error
  ) {
    super(message);
    this.name = 'AppError';
  }
}

interface RetryConfig {
  maxRetries: number;
  initialDelay: number;
  maxDelay: number;
}

export class ApiClient {
  private client: AxiosInstance;
  private retryConfig: RetryConfig = {
    maxRetries: 3,
    initialDelay: 1000,
    maxDelay: 10000,
  };

  constructor(baseURL: string) {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(this.handleError(error));
      }
    );

    // Response interceptor with retry logic
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (this.shouldRetry(error)) {
          return this.retryRequest(error.config);
        }
        return Promise.reject(this.handleError(error));
      }
    );
  }

  private shouldRetry(error: AxiosError): boolean {
    if (!error.config) return false;
    
    const retryCount = (error.config as any).__retryCount || 0;
    if (retryCount >= this.retryConfig.maxRetries) return false;

    // Retry on network errors and 5xx status codes
    if (!error.response) return true;
    return error.response.status >= 500;
  }

  private async retryRequest(config: AxiosRequestConfig) {
    const retryCount = ((config as any).__retryCount || 0) + 1;
    (config as any).__retryCount = retryCount;

    const delay = Math.min(
      this.retryConfig.initialDelay * Math.pow(2, retryCount - 1),
      this.retryConfig.maxDelay
    );

    await new Promise((resolve) => setTimeout(resolve, delay));
    return this.client(config);
  }

  private handleError(error: AxiosError | Error): AppError {
    if (error instanceof AxiosError) {
      const statusCode = error.response?.status || 500;
      const message = (error.response?.data as any)?.detail || error.message;
      const code = (error.response?.data as any)?.error_code || 'NETWORK_ERROR';

      return new AppError(code, message, statusCode, error);
    }

    return new AppError('UNKNOWN_ERROR', error.message, 500, error as Error);
  }

  async get<T>(url: string, schema?: z.ZodSchema): Promise<T> {
    try {
      const response = await this.client.get<T>(url);
      if (schema) {
        return schema.parse(response.data);
      }
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError | Error);
    }
  }

  async post<T>(url: string, data?: any, schema?: z.ZodSchema): Promise<T> {
    try {
      const response = await this.client.post<T>(url, data);
      if (schema) {
        return schema.parse(response.data);
      }
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError | Error);
    }
  }

  async put<T>(url: string, data?: any, schema?: z.ZodSchema): Promise<T> {
    try {
      const response = await this.client.put<T>(url, data);
      if (schema) {
        return schema.parse(response.data);
      }
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError | Error);
    }
  }

  async patch<T>(url: string, data?: any, schema?: z.ZodSchema): Promise<T> {
    try {
      const response = await this.client.patch<T>(url, data);
      if (schema) {
        return schema.parse(response.data);
      }
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError | Error);
    }
  }

  async delete<T>(url: string, schema?: z.ZodSchema): Promise<T> {
    try {
      const response = await this.client.delete<T>(url);
      if (schema) {
        return schema.parse(response.data);
      }
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError | Error);
    }
  }
}

const apiClient = new ApiClient(
  (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000/api'
);

export default apiClient;
