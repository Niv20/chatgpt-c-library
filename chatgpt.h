/*
 * ChatGPT API Library for C
 * A comprehensive C library for interacting with OpenAI's Chat Completions API
 * 
 * Features:
 * - Multi-turn conversation management
 * - Streaming and non-streaming responses
 * - Comprehensive error handling
 * - Flexible configuration options
 * - Conversation persistence (save/load)
 * - Token usage tracking
 * - Global API key management
 * 
 * Copyright (c) 2025
 * Licensed under MIT License
 */

#ifndef CHATGPT_H
#define CHATGPT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdio.h>

/*
 * Simple library for working with OpenAI Chat Completions API in C.
 * Goal: Enable conversation building, regular or streaming requests, configuration, and error handling.
 * Most functions return 0 on success and non-zero on failure (with specific error codes).
 */

/* ========== ERROR CODES ========== */

/**
 * Error codes returned by library functions
 * All functions return CHATGPT_OK (0) on success, or one of these error codes on failure
 */
typedef enum {
    CHATGPT_OK,               // Success
    CHATGPT_ERR_OOM,          // Out of memory
    CHATGPT_ERR_INVALID_ARG,  // Invalid argument provided
    CHATGPT_ERR_HTTP,         // HTTP/network error
    CHATGPT_ERR_JSON_PARSE,   // JSON parsing failed
    CHATGPT_ERR_API,          // API returned an error
    CHATGPT_ERR_STREAM,       // Streaming error
    CHATGPT_ERR_STATE         // Invalid internal state
} ChatGPT_ErrorCode;

/* ========== DATA STRUCTURES ========== */

/**
 * Represents a single message in a conversation
 * Each message has a role (user, assistant, system) and content (the actual text)
 */
typedef struct {
    char *role;     // Message role: "user", "assistant", or "system"
    char *content;  // Message content (the actual text)
} ChatGPTMessage;

/**
 * Token usage information from API responses
 * Tracks how many tokens were used for prompt, completion, and total
 */
typedef struct {
    int prompt_tokens;      // Tokens used in the prompt
    int completion_tokens;  // Tokens used in the completion
    int total_tokens;       // Total tokens used (prompt + completion)
} ChatGPTUsage;

/**
 * Main conversation structure for managing ChatGPT interactions
 * Contains configuration, conversation history, and state information
 */
typedef struct ChatGPTConversation {
    // Configuration
    char *api_key;              // OpenAI API key (private copy)
    char *model;                // Model name (e.g., "gpt-4", "gpt-3.5-turbo")
    double temperature;         // Creativity/randomness (0.0 to 2.0)
    double top_p;              // Nucleus sampling parameter (0.0 to 1.0)
    int max_tokens;            // Maximum tokens for completion (0 = no limit)
    double presence_penalty;    // Penalty for token presence (-2.0 to 2.0)
    double frequency_penalty;   // Penalty for token frequency (-2.0 to 2.0)
    char *base_url;            // API base URL (for custom endpoints)
    
    // New streaming and context configuration
    int use_streaming;          // 1 = streaming mode (default), 0 = complete response
    int context_messages;       // Number of recent messages to send to API (default: 5, 0 = only last message)
    
    // Retry configuration
    int max_retries;           // Maximum number of retry attempts (default: 3)
    int retry_delay_ms;        // Delay between retries in milliseconds (default: 1000)

    // Conversation state
    ChatGPTMessage *messages;   // Dynamic array of messages
    size_t message_count;       // Number of messages currently stored
    size_t message_capacity;    // Allocated capacity for messages array

    // Response tracking
    ChatGPTUsage last_usage;    // Token usage from last API call
    char *last_reply;          // Complete response from last API call

    // Error handling
    char last_error[512];       // Last error message text
    ChatGPT_ErrorCode last_code;// Last error code
    long last_http_code;       // Last HTTP response code
} ChatGPTConversation;

// Alias for backward compatibility
typedef ChatGPTConversation ChatGPTClient;

/* ========== GLOBAL API KEY MANAGEMENT ========== */

/**
 * Set a global API key that can be reused across multiple clients
 * Once set, you can pass NULL to client creation functions to use this global key
 */
int chatgpt_set_api_key_global(const char *api_key);

/**
 * Get the currently set global API key
 * Returns NULL if no global key has been set
 */
const char *chatgpt_get_api_key_global(void);

/**
 * Set a log file for debugging output
 * Pass NULL to disable logging. The library will not close this file.
 */
int chatgpt_set_log_file(FILE *f);

/* ========== CONVERSATION LIFECYCLE ========== */

/**
 * Create a new ChatGPT conversation instance
 * api_key: OpenAI API key, or NULL to use global key
 * model: Model name (e.g., "gpt-4", "gpt-3.5-turbo"), or NULL for default
 * Returns: New conversation instance or NULL on error
 */
ChatGPTConversation *chatgpt_conversation_new(const char *api_key, const char *model);

/**
 * Free all resources associated with a ChatGPT conversation
 * Must be called for every conversation created with chatgpt_conversation_new()
 */
void chatgpt_conversation_free(ChatGPTConversation *conversation);

/**
 * Copy all settings from one conversation to another
 * Does not copy messages, only configuration settings
 */
int chatgpt_conversation_copy_settings(ChatGPTConversation *dest, const ChatGPTConversation *src);

/**
 * Create a new ChatGPT client instance (legacy compatibility)
 * api_key: OpenAI API key, or NULL to use global key
 * model: Model name (e.g., "gpt-4", "gpt-3.5-turbo"), or NULL for default
 * Returns: New client instance or NULL on error
 */
ChatGPTClient *chatgpt_client_new(const char *api_key, const char *model);

/**
 * Free all resources associated with a ChatGPT client (legacy compatibility)
 * Must be called for every client created with chatgpt_client_new()
 */
void chatgpt_client_free(ChatGPTClient *client);

/* ========== ERROR HANDLING ========== */

/**
 * Clear the last error state
 * Resets error code to OK and clears error message
 */
void chatgpt_clear_error(ChatGPTConversation *conversation);

/**
 * Get the last error message as a string
 * Returns empty string if no error occurred
 */
const char *chatgpt_last_error(const ChatGPTConversation *conversation);

/**
 * Get the last error code
 * Returns CHATGPT_OK if no error occurred
 */
ChatGPT_ErrorCode chatgpt_last_code(const ChatGPTConversation *conversation);

/**
 * Get the last HTTP response code
 * Useful for identifying rate limits (429) and other HTTP-specific errors
 */
long chatgpt_last_http_code(const ChatGPTConversation *conversation);

/* ========== CONFIGURATION ========== */

/**
 * Set the AI model to use for completions
 * Common models: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o-mini"
 */
int chatgpt_set_model(ChatGPTConversation *conversation, const char *model);

/**
 * Set the temperature parameter (controls randomness/creativity)
 * Range: 0.0 to 2.0 (0.0 = deterministic, 2.0 = very creative)
 */
int chatgpt_set_temperature(ChatGPTConversation *conversation, double temperature);

/**
 * Set the top_p parameter (nucleus sampling)
 * Range: 0.0 to 1.0 (controls diversity of token selection)
 */
int chatgpt_set_top_p(ChatGPTConversation *conversation, double top_p);

/**
 * Set the presence penalty parameter
 * Range: -2.0 to 2.0 (default: 0.0)
 * Positive values penalize tokens that have already appeared in the text so far,
 * encouraging the model to introduce new topics and avoid repetition.
 * - 0.0: No penalty (default)
 * - Positive values: Discourage repetition of topics/words that already appeared
 * - Negative values: Encourage repetition of topics/words that already appeared
 * Use Case: Set to positive value (e.g., 0.6) to encourage diverse, creative content
 */
int chatgpt_set_presence_penalty(ChatGPTConversation *conversation, double presence_penalty);

/**
 * Set the frequency penalty parameter  
 * Range: -2.0 to 2.0 (default: 0.0)
 * Positive values penalize tokens based on their frequency in the text so far,
 * with higher penalties for more frequently used words.
 * - 0.0: No penalty (default)
 * - Positive values: Reduce repetitive word usage proportionally to frequency
 * - Negative values: Encourage repetitive word usage proportionally to frequency
 * Use Case: Set to positive value (e.g., 0.3) to reduce word repetition while allowing natural flow
 */
int chatgpt_set_frequency_penalty(ChatGPTConversation *conversation, double frequency_penalty);

/**
 * Set maximum tokens for the completion
 * 0 = no limit, positive number = maximum tokens to generate
 */
int chatgpt_set_max_tokens(ChatGPTConversation *conversation, int max_tokens);

/**
 * Set the base URL for API requests
 * Default: "https://api.openai.com" (for custom/self-hosted endpoints)
 */
int chatgpt_set_base_url(ChatGPTConversation *conversation, const char *base_url);

/**
 * Enable or disable streaming mode
 * 1 = streaming mode (default), 0 = complete response at once
 */
int chatgpt_set_streaming(ChatGPTConversation *conversation, int use_streaming);

/**
 * Set the number of recent messages to include in API requests
 * 0 = only last message, positive number = number of recent messages
 * Default: 5
 */
int chatgpt_set_context_messages(ChatGPTConversation *conversation, int context_messages);

/**
 * Set retry configuration for failed requests
 * max_retries: Maximum number of retry attempts (default: 3)
 * delay_ms: Delay between retries in milliseconds (default: 1000)
 */
int chatgpt_set_retry_config(ChatGPTConversation *conversation, int max_retries, int delay_ms);

/* ========== MESSAGE MANAGEMENT ========== */

/**
 * Clear all messages from the conversation
 * Removes all messages but keeps configuration settings
 */
int chatgpt_clear_messages(ChatGPTConversation *conversation);

/**
 * Add a message to the conversation with specified role and content
 * role: "user", "assistant", or "system"
 * content: The message text
 */
int chatgpt_add_message(ChatGPTConversation *conversation, const char *role, const char *content);

/**
 * Add a user message to the conversation
 * Convenience function equivalent to chatgpt_add_message(conversation, "user", content)
 */
int chatgpt_add_user(ChatGPTConversation *conversation, const char *content);

/**
 * Add a system message to the conversation
 * System messages set the behavior and context for the AI
 */
int chatgpt_add_system(ChatGPTConversation *conversation, const char *content);

/**
 * Add an assistant message to the conversation
 * Useful for providing examples or continuing a conversation
 */
int chatgpt_add_assistant(ChatGPTConversation *conversation, const char *content);

/**
 * Add a user message with an attached file (image or document)
 * file_path: Path to the file to attach
 * file_type: "image" for images, "document" for other files
 * content: Optional text content to accompany the file
 */
int chatgpt_add_user_with_file(ChatGPTConversation *conversation, const char *content, 
                               const char *file_path, const char *file_type);

/**
 * Get the number of messages in the conversation
 */
int chatgpt_get_message_count(const ChatGPTConversation *conversation);

/**
 * Remove the last message from the conversation
 */
int chatgpt_pop_last_message(ChatGPTConversation *conversation);

/**
 * Remove a message at a specific index (0-based)
 */
int chatgpt_remove_message_at(ChatGPTConversation *conversation, size_t index);

/**
 * Replace the content of the last user message
 * Searches backwards to find the most recent user message
 */
int chatgpt_replace_last_user(ChatGPTConversation *conversation, const char *new_content);

/**
 * Append text to the last assistant message
 * Useful for building up responses incrementally
 */
int chatgpt_append_to_last_assistant(ChatGPTConversation *conversation, const char *extra_text);

/**
 * Reset the conversation to a clean state
 * Clears messages, usage statistics, last reply, and errors
 * Keeps configuration settings (model, temperature, etc.)
 */
int chatgpt_reset(ChatGPTConversation *conversation);

/* ========== API COMMUNICATION ========== */

/**
 * Send a chat completion request and get the full response
 * This is the main function for getting AI responses
 * Returns: Complete AI response as a new string (caller must free), or NULL on error
 */
char *chatgpt_chat_complete(ChatGPTConversation *conversation);

/**
 * Callback function type for streaming responses
 * delta: Small piece of response text
 * user_data: User-provided data passed to the callback
 */
typedef void (*chatgpt_stream_callback)(const char *delta, void *user_data);

/**
 * Send a streaming chat completion request
 * Calls the provided callback function for each chunk of response text
 * cb: Callback function (can be NULL)
 * user_data: Data passed to callback
 * full_response_out: Pointer to receive complete response (can be NULL)
 */
int chatgpt_chat_complete_stream(ChatGPTConversation *conversation,
                                chatgpt_stream_callback cb,
                                void *user_data,
                                char **full_response_out);

/**
 * Simple one-shot query function (legacy compatibility)
 * Creates a temporary client, sends a single user message, and returns the response
 */
char *chatgpt_query(const char *api_key, const char *prompt);

/**
 * Get usage statistics from the last API call
 * Returns token counts for prompt, completion, and total usage
 */
int chatgpt_get_last_usage(ChatGPTConversation *conversation, ChatGPTUsage *usage_out);

/**
 * Get the last reply from the AI (cached)
 * Returns the complete response from the most recent API call
 */
const char *chatgpt_last_reply(ChatGPTConversation *conversation);

/**
 * Get a list of available models from the API
 * Returns a JSON string with available models (caller must free)
 */
char *chatgpt_get_available_models(const char *api_key);

/**
 * Check if a specific model is available
 * Returns 1 if available, 0 if not available, -1 on error
 */
int chatgpt_is_model_available(const char *api_key, const char *model_name);

/* ========== CONVERSATION PERSISTENCE ========== */

/**
 * Save the current conversation to a JSON file
 * Saves only the messages array, not configuration settings
 */
int chatgpt_save_conversation(ChatGPTConversation *conversation, const char *path);

/**
 * Load a conversation from a JSON file
 * Replaces current messages with those from the file
 */
int chatgpt_load_conversation(ChatGPTConversation *conversation, const char *path);

/* ========== UTILITY FUNCTIONS ========== */

/**
 * Create a JSON representation of all messages in the conversation
 * Returns JSON string (caller must free) or NULL on error
 */
char *chatgpt_build_messages_json(ChatGPTConversation *conversation);

/**
 * Create a JSON dump of all messages (alias for chatgpt_build_messages_json)
 */
char *chatgpt_dump_messages(ChatGPTConversation *conversation);

/**
 * Print all messages to a file stream for debugging
 */
void chatgpt_print_messages(ChatGPTConversation *conversation, FILE *out);

/**
 * Remove trailing whitespace from a string
 * Modifies the string in-place
 */
void chatgpt_rtrim(char *s);

/**
 * Generate an image using DALL-E
 * prompt: Description of the image to generate
 * size: Image size ("1024x1024", "512x512", "256x256")
 * Returns: URL to the generated image (caller must free) or NULL on error
 */
char *chatgpt_generate_image(const char *api_key, const char *prompt, const char *size);


#ifdef __cplusplus
}
#endif

#endif /* CHATGPT_H */
