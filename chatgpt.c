/*
 * ChatGPT API Library for C
 * A clean and simple implementation for interacting with OpenAI's Chat Completions API
 * Supports conversation management, streaming responses, error handling, and configuration
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include "cJSON.h"
#include "chatgpt.h"

#define DEFAULT_MODEL "gpt-4o-mini"

/*
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃               GLOBAL VARIABLES                ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

// Global API key that can be set once and reused across multiple clients
static char *g_api_key = NULL; 

// Optional log file for debugging purposes
static FILE *g_log = NULL;          

/*
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃               HELPER FUNCTIONS                ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * String duplication helper function
 * Creates a dynamically allocated copy of the input string
 * Usage: char *copy = dup_str("hello"); 
 */
static char *dup_str(const char *s) {

    if (!s) return NULL;
    
    // Calculate string length
    size_t len = strlen(s);

    // Allocate memory (+1 because we need space for the null terminator)
    char *p = (char*)malloc(len + 1);

    // Edge case: return NULL if memory allocation fails
    if (!p) return NULL;
    
    // Copy string including null terminator
    memcpy(p, s, len + 1);

    // Return the duplicated string
    return p;
}

/*
 * Set a log file for debugging output
 * The library will write debug information to this file
 * Usage: chatgpt_set_log_file(fopen("debug.log", "w"));
 * Returns: CHATGPT_OK (always succeeds)
 */
int chatgpt_set_log_file(FILE *f) {
    g_log = f;
    return CHATGPT_OK;
}

/*
 * Simple logging function
 * Writes a message to the global log file, if one is set
 * Usage: log_line("API request sent"); 
 */
static void log_line(const char *m) {

    if (!g_log) return;

    // If we got here, the file is set, so we can log the message

    // Writes the string 'm' to the file 'g_log' (without automatically adding a newline)
    fputs(m, g_log);

    // Writes a single newline character ('\n') to the file 'g_log'
    fputc('\n', g_log);

    // Forces any buffered output data for 'g_log' to be written immediately to the file
    fflush(g_log);
}

/*
 * Error handling helper function
 * Sets the error code and message on a ChatGPT client
 * Usage: set_error(client, CHATGPT_ERR_HTTP, "Connection failed");
 */
/*
 * Error handling helper function
 * Sets the error code and message on a ChatGPT conversation
 * Usage: set_error(conversation, CHATGPT_ERR_HTTP, "Connection failed");
 */
static void set_error(ChatGPTConversation *c, ChatGPT_ErrorCode code, const char *msg) {

    if (!c) return;
    
    // Set error code
    c->last_code = code;
    
    if (msg) {
        strncpy(c->last_error, msg, sizeof(c->last_error) - 1);
        c->last_error[sizeof(c->last_error) - 1] = '\0';
    } else {
        c->last_error[0] = '\0';
    }
}

/*
 * Dynamic array capacity management
 * Ensures the messages array has enough capacity for the required number of elements
 * Doubles capacity when expansion is needed 
 * Usage: Internal function, called automatically when adding messages
 */
static int ensure_cap(ChatGPTConversation *c, size_t need) {

    // Check if we already have enough capacity
    if (need <= c->message_capacity) return CHATGPT_OK;

    // Calculate new capacity (start with 4, then double each time)
    size_t cap = c->message_capacity ? c->message_capacity * 2 : 4;
    
    // Doubles the capacity until it is large enough to accommodate the new messages. 
    // This is a crucial safety measure to handle cases where multiple messages might be added at once, 
    // ensuring the array always has sufficient space without requiring multiple, less-efficient reallocations. 
    // This strategy optimizes for common single-message additions while providing robustness for more complex operations.
    while (cap < need) cap *= 2; 
    
    // Reallocate memory for messages array
    ChatGPTMessage *m = (ChatGPTMessage*)realloc(c->messages, cap * sizeof(ChatGPTMessage));
    if (!m) return CHATGPT_ERR_OOM;
    
    // Initialize new slots to NULL
    for (size_t i = c->message_capacity; i < cap; i++) {
        m[i].role = NULL;
        m[i].content = NULL;
    }
    
    // Update client structure
    c->messages = m;    // In case of reallocation
    c->message_capacity = cap;
    return CHATGPT_OK;
}

/*
 * Remove trailing whitespace from a string
 * Modifies the string in-place by null-terminating at the first trailing whitespace
 * Useful for cleaning up response text
 * Usage: 
 *   char text[] = "Hello world   \n\r\t";
 *   chatgpt_rtrim(text);  // Now text is "Hello world"
 * Parameters:
 *   - s: String to trim (modified in-place)
 */
void chatgpt_rtrim(char *s) {
    if (!s) return;
    
    size_t n = strlen(s);
    
    // Work backwards from end of string
    while (n > 0) {
        unsigned char ch = (unsigned char)s[n - 1];
        
        // Check if character is whitespace
        if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t') {
            s[n - 1] = '\0';  // Remove whitespace character
            --n;
        } else {
            break;  // Found non-whitespace, stop trimming
        }
    }
}

/*
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃           GLOBAL API KEY MANAGEMENT           ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * Set a global API key that can be reused across multiple clients
 * This allows you to set the key once and pass NULL to client creation functions
 * Usage: chatgpt_set_api_key_global("your-openai-api-key-here");
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_set_api_key_global(const char *k) {

    if (!k) return CHATGPT_ERR_INVALID_ARG;
    
    // Create a copy of the key
    char *d = dup_str(k);
    if (!d) return CHATGPT_ERR_OOM;
    
    // Free old key and set new one
    free(g_api_key);
    g_api_key = d;
    return CHATGPT_OK;
}

/*
 * Get the currently set global API key
 * Usage: const char *key = chatgpt_get_api_key_global();
 * Returns: Pointer to global key or NULL if not set (do not free this pointer)
 */
const char *chatgpt_get_api_key_global(void) {
    return g_api_key;
}


/*
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃             JSON MESSAGE BUILDING             ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * Build a JSON array representation of all messages in the conversation
 * This function creates a JSON array suitable for the OpenAI API
 * Usage: char *json = chatgpt_build_messages_json(conversation); ... free(json);
 * Returns: JSON string or NULL on error
 */
char *chatgpt_build_messages_json(ChatGPTConversation *c) {
    
    if (!c) return NULL;
    
    // Create JSON array for messages
    cJSON *arr = cJSON_CreateArray();
    if (!arr) return NULL;
    
    // Add each message to the array
    for (size_t i = 0; i < c->message_count; i++) {
        // Create message object
        cJSON *o = cJSON_CreateObject();
        if (!o) {
            cJSON_Delete(arr);
            return NULL;
        }
        
        // Add role and content fields
        cJSON_AddStringToObject(o, "role", c->messages[i].role ? c->messages[i].role : "user");
        cJSON_AddStringToObject(o, "content", c->messages[i].content ? c->messages[i].content : "");
        
        // Add message to array
        cJSON_AddItemToArray(arr, o);
    }
    
    // Convert to string and cleanup
    char *s = cJSON_PrintUnformatted(arr);
    cJSON_Delete(arr);
    return s;
}


/*
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃        CONVERSATION LIFECYCLE MANAGEMENT      ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * Create a new ChatGPT conversation instance
 * This is the main constructor for the library - creates and initializes a conversation
 * Usage: ChatGPTConversation *conv = chatgpt_conversation_new("sk-your-key", "gpt-4");
 *        ChatGPTConversation *conv = chatgpt_conversation_new(NULL, NULL); // Uses global key and default model
 * Parameters:
 *   - api_key: OpenAI API key, or NULL to use global key
 *   - model: Model name (e.g., "gpt-4", "gpt-3.5-turbo"), or NULL for default
 * Returns: New conversation instance or NULL on error
 */
ChatGPTConversation *chatgpt_conversation_new(const char *api_key, const char *model) {
    ChatGPTConversation *c;
    
    // Use global key if none provided
    if (!api_key) api_key = g_api_key;
    if (!api_key) return NULL;  // No key available
    
    // Allocate and initialize conversation structure
    c = (ChatGPTConversation*)calloc(1, sizeof(ChatGPTConversation));
    if (!c) return NULL;
    
    // Copy API key and model
    c->api_key = dup_str(api_key);
    c->model = dup_str(model ? model : DEFAULT_MODEL); 
    
    // Set default parameters
    c->temperature = 0.7;       // Balanced creativity
    c->top_p = 1.0;            // No nucleus sampling by default
    c->max_tokens = 0;         // No token limit by default
    c->presence_penalty = 0.0;  // No presence penalty by default
    c->frequency_penalty = 0.0; // No frequency penalty by default
    
    // Set new default parameters
    c->use_streaming = 1;      // Streaming enabled by default
    c->context_messages = 5;   // Send last 5 messages by default
    c->max_retries = 3;        // 3 retry attempts by default
    c->retry_delay_ms = 1000;  // 1 second delay between retries
    
    // Set default base URL for OpenAI API
    c->base_url = dup_str("https://api.openai.com");
    
    // Initialize other fields
    c->last_reply = NULL;
    c->last_error[0] = '\0';
    c->last_code = CHATGPT_OK;
    c->last_http_code = 0;
    
    return c;
}

/*
 * Free all resources associated with a ChatGPT conversation
 * This cleans up all allocated memory and should be called for every conversation
 * Usage: chatgpt_conversation_free(conversation); conversation = NULL; // Good practice to null the pointer
 */
void chatgpt_conversation_free(ChatGPTConversation *c) {
    if (!c) return;
    
    // Free string fields
    free(c->api_key);
    free(c->model);
    free(c->base_url);
    free(c->last_reply);
    
    // Free all messages
    for (size_t i = 0; i < c->message_count; i++) {
        free(c->messages[i].role);
        free(c->messages[i].content);
    }
    free(c->messages);
    
    // Free the conversation structure itself
    free(c);
}

/*
 * Copy all settings from one conversation to another
 * Does not copy messages, only configuration settings
 * Usage: chatgpt_conversation_copy_settings(dest_conv, src_conv);
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_conversation_copy_settings(ChatGPTConversation *dest, const ChatGPTConversation *src) {
    if (!dest || !src) return CHATGPT_ERR_INVALID_ARG;
    
    // Copy model
    if (src->model) {
        char *new_model = dup_str(src->model);
        if (!new_model) return CHATGPT_ERR_OOM;
        free(dest->model);
        dest->model = new_model;
    }
    
    // Copy base URL
    if (src->base_url) {
        char *new_url = dup_str(src->base_url);
        if (!new_url) return CHATGPT_ERR_OOM;
        free(dest->base_url);
        dest->base_url = new_url;
    }
    
    // Copy numerical settings
    dest->temperature = src->temperature;
    dest->top_p = src->top_p;
    dest->max_tokens = src->max_tokens;
    dest->presence_penalty = src->presence_penalty;
    dest->frequency_penalty = src->frequency_penalty;
    dest->use_streaming = src->use_streaming;
    dest->context_messages = src->context_messages;
    dest->max_retries = src->max_retries;
    dest->retry_delay_ms = src->retry_delay_ms;
    
    return CHATGPT_OK;
}

/*
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃          CLIENT LIFECYCLE MANAGEMENT          ┃
┃          (Legacy Compatibility)               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * Create a new ChatGPT client instance
 * This is the main constructor for the library - creates and initializes a client
 * Usage: ChatGPTClient *client = chatgpt_client_new("sk-your-key", "gpt-4");
 *        ChatGPTClient *client = chatgpt_client_new(NULL, NULL); // Uses global key and default model
 * Parameters:
 *   - api_key: OpenAI API key, or NULL to use global key
 *   - model: Model name (e.g., "gpt-4", "gpt-3.5-turbo"), or NULL for default
 * Returns: New client instance or NULL on error
 */
/*
 * Create a new ChatGPT client instance (legacy compatibility)
 * This is an alias for the new conversation function
 * Usage: ChatGPTClient *client = chatgpt_client_new("sk-your-key", "gpt-4");
 *        ChatGPTClient *client = chatgpt_client_new(NULL, NULL); // Uses global key and default model
 * Parameters:
 *   - api_key: OpenAI API key, or NULL to use global key
 *   - model: Model name (e.g., "gpt-4", "gpt-3.5-turbo"), or NULL for default
 * Returns: New client instance or NULL on error
 */
ChatGPTClient *chatgpt_client_new(const char *api_key, const char *model) {
    return chatgpt_conversation_new(api_key, model);
}

/*
 * Free all resources associated with a ChatGPT client (legacy compatibility)
 * This cleans up all allocated memory and should be called for every client
 * Usage: chatgpt_client_free(client); client = NULL; // Good practice to null the pointer
 */
void chatgpt_client_free(ChatGPTClient *c) {
    chatgpt_conversation_free(c);
}

/*
 * Clear the last error state
 * Resets error code to OK and clears error message
 * Usage: chatgpt_clear_error(conversation); // After handling an error
 */
void chatgpt_clear_error(ChatGPTConversation *c) {
    if (c) {
        c->last_error[0] = '\0';
        c->last_code = CHATGPT_OK;
        c->last_http_code = 0;
    }
}

/*
 * Get the last error message as a string
 * Usage: if (result != CHATGPT_OK) printf("Error: %s\n", chatgpt_last_error(conversation));
 * Returns: Error message string (empty if no error, do not free this pointer)
 */
const char *chatgpt_last_error(const ChatGPTConversation *c) {
    return c ? c->last_error : "";
}

/*
 * Get the last error code
 * Usage: ChatGPT_ErrorCode code = chatgpt_last_code(conversation);
 * Returns: Error code enum value
 */
ChatGPT_ErrorCode chatgpt_last_code(const ChatGPTConversation *c) {
    return c ? c->last_code : CHATGPT_ERR_INVALID_ARG;
}

/*
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃           CONVERSATION CONFIGURATION          ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * Set the AI model to use for completions
 * Common models: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o-mini"
 * Usage: chatgpt_set_model(conversation, "gpt-4");
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_set_model(ChatGPTConversation *c, const char *m) {
    if (!c || !m) return CHATGPT_ERR_INVALID_ARG;
    
    // Create copy of new model name
    char *d = dup_str(m);
    if (!d) return CHATGPT_ERR_OOM;
    
    // Replace old model with new one
    free(c->model);
    c->model = d;
    return CHATGPT_OK;
}

/*
 * Set the temperature parameter (controls randomness/creativity)
 * Range: 0.0 to 2.0
 * - 0.0: Very deterministic, same input → same output
 * - 1.0: Balanced creativity
 * - 2.0: Very creative/random
 * Usage: chatgpt_set_temperature(conversation, 0.7);
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_set_temperature(ChatGPTConversation *c, double v) {
    if (!c) return CHATGPT_ERR_INVALID_ARG;
    if (v < 0 || v > 2) return CHATGPT_ERR_INVALID_ARG;
    
    c->temperature = v;
    return CHATGPT_OK;
}

/*
 * Set the top_p parameter (nucleus sampling)
 * Range: 0.0 to 1.0 (exclusive of 0.0)
 * Controls diversity by considering only the top percentage of probable tokens
 * - 0.1: Very focused, only most likely tokens
 * - 1.0: Consider all tokens
 * Usage: chatgpt_set_top_p(conversation, 0.9);
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_set_top_p(ChatGPTConversation *c, double v) {
    if (!c) return CHATGPT_ERR_INVALID_ARG;
    if (v <= 0 || v > 1) return CHATGPT_ERR_INVALID_ARG;
    
    c->top_p = v;
    return CHATGPT_OK;
}

/*
 * Set the presence penalty parameter
 * Range: -2.0 to 2.0 (default: 0.0)
 * Positive values penalize tokens that have already appeared in the text so far,
 * encouraging the model to introduce new topics and avoid repetition.
 * - 0.0: No penalty (default)
 * - Positive values: Discourage repetition of topics/words that already appeared
 * - Negative values: Encourage repetition of topics/words that already appeared
 * 
 * Key behavior: Penalizes ANY occurrence of a word/topic that already appeared.
 * If a word appeared once, the model will be "reluctant" to repeat it.
 * This causes the model to introduce new ideas to the text.
 * 
 * Usage: chatgpt_set_presence_penalty(conversation, 0.6); // Encourage diverse content
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_set_presence_penalty(ChatGPTConversation *c, double v) {
    if (!c) return CHATGPT_ERR_INVALID_ARG;
    if (v < -2.0 || v > 2.0) return CHATGPT_ERR_INVALID_ARG;
    
    c->presence_penalty = v;
    return CHATGPT_OK;
}

/*
 * Set the frequency penalty parameter  
 * Range: -2.0 to 2.0 (default: 0.0)
 * Positive values penalize tokens based on their frequency in the text so far,
 * with higher penalties for more frequently used words.
 * - 0.0: No penalty (default)
 * - Positive values: Reduce repetitive word usage proportionally to frequency
 * - Negative values: Encourage repetitive word usage proportionally to frequency
 * 
 * Key behavior: Penalizes repetition based on HOW MANY TIMES it occurred.
 * If a word repeats many times, the penalty grows.
 * This doesn't prevent repetition completely, but spreads word usage more evenly.
 * 
 * Usage: chatgpt_set_frequency_penalty(conversation, 0.3); // Reduce word repetition
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_set_frequency_penalty(ChatGPTConversation *c, double v) {
    if (!c) return CHATGPT_ERR_INVALID_ARG;
    if (v < -2.0 || v > 2.0) return CHATGPT_ERR_INVALID_ARG;
    
    c->frequency_penalty = v;
    return CHATGPT_OK;
}

/*
 * Set maximum tokens for the completion
 * Limits the length of the AI's response
 * - 0: No limit (uses model's default)
 * - Positive number: Maximum tokens to generate
 * Usage: chatgpt_set_max_tokens(conversation, 150); // Limit to 150 tokens
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_set_max_tokens(ChatGPTConversation *c, int n) {
    if (!c || n < 0) return CHATGPT_ERR_INVALID_ARG;
    
    c->max_tokens = n;
    return CHATGPT_OK;
}

/*
 * Set the base URL for API requests
 * Allows using alternative OpenAI-compatible endpoints
 * Default: "https://api.openai.com"
 * Usage: chatgpt_set_base_url(conversation, "https://my-custom-api.com");
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_set_base_url(ChatGPTConversation *c, const char *u) {
    if (!c || !u) return CHATGPT_ERR_INVALID_ARG;
    
    // Create copy of new URL
    char *d = dup_str(u);
    if (!d) return CHATGPT_ERR_OOM;
    
    // Replace old URL with new one
    free(c->base_url);
    c->base_url = d;
    return CHATGPT_OK;
}

/*
 * Enable or disable streaming mode
 * 1 = streaming mode (default), 0 = complete response at once
 * Usage: chatgpt_set_streaming(conversation, 1); // Enable streaming
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_set_streaming(ChatGPTConversation *c, int use_streaming) {
    if (!c) return CHATGPT_ERR_INVALID_ARG;
    
    c->use_streaming = use_streaming ? 1 : 0;
    return CHATGPT_OK;
}

/*
 * Set the number of recent messages to include in API requests
 * 0 = only last message, positive number = number of recent messages
 * Default: 5
 * Usage: chatgpt_set_context_messages(conversation, 10); // Send last 10 messages
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_set_context_messages(ChatGPTConversation *c, int context_messages) {
    if (!c || context_messages < 0) return CHATGPT_ERR_INVALID_ARG;
    
    c->context_messages = context_messages;
    return CHATGPT_OK;
}

/*
 * Set retry configuration for failed requests
 * max_retries: Maximum number of retry attempts (default: 3)
 * delay_ms: Delay between retries in milliseconds (default: 1000)
 * Usage: chatgpt_set_retry_config(conversation, 5, 2000); // 5 retries, 2 second delay
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_set_retry_config(ChatGPTConversation *c, int max_retries, int delay_ms) {
    if (!c || max_retries < 0 || delay_ms < 0) return CHATGPT_ERR_INVALID_ARG;
    
    c->max_retries = max_retries;
    c->retry_delay_ms = delay_ms;
    return CHATGPT_OK;
}

/*
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃              MESSAGE MANAGEMENT               ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * Add a message to the conversation with specified role and content
 * This is the core function for building conversations
 * Usage: chatgpt_add_message(client, "user", "Hello, how are you?");
 *        chatgpt_add_message(client, "assistant", "I'm doing well, thank you!");
 * Parameters:
 *   - role: "user", "assistant", or "system"
 *   - content: The message text
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_add_message(ChatGPTConversation *c, const char *role, const char *content) {
    int r;
    char *r1, *c1;
    
    if (!c || !role || !content) return CHATGPT_ERR_INVALID_ARG;
    
    // Ensure we have capacity for one more message
    r = ensure_cap(c, c->message_count + 1);
    if (r) return r;
    
    // Create copies of role and content strings
    r1 = dup_str(role);
    c1 = dup_str(content);
    if (!r1 || !c1) {
        free(r1);
        free(c1);
        return CHATGPT_ERR_OOM;
    }
    
    // Add message to array
    c->messages[c->message_count].role = r1;
    c->messages[c->message_count].content = c1;
    c->message_count++;
    
    return CHATGPT_OK;
}

/*
 * Add a user message to the conversation
 * Convenience function equivalent to chatgpt_add_message(conversation, "user", text)
 * Usage: chatgpt_add_user(conversation, "What's the weather like today?");
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_add_user(ChatGPTConversation *c, const char *t) {
    return chatgpt_add_message(c, "user", t);
}

/*
 * Add a system message to the conversation
 * System messages set the behavior and context for the AI
 * Usage: chatgpt_add_system(client, "You are a helpful coding assistant.");
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_add_system(ChatGPTClient *c, const char *t) {
    return chatgpt_add_message(c, "system", t);
}

/*
 * Add an assistant message to the conversation
 * Useful for providing examples or continuing a conversation
 * Usage: chatgpt_add_assistant(client, "I'd be happy to help with that!");
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_add_assistant(ChatGPTClient *c, const char *t) {
    return chatgpt_add_message(c, "assistant", t);
}

/*
 * Clear all messages from the conversation
 * Removes all messages but keeps configuration settings
 * Usage: chatgpt_clear_messages(client); // Start fresh conversation
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_clear_messages(ChatGPTClient *c) {
    if (!c) return CHATGPT_ERR_INVALID_ARG;
    
    // Free all message content
    for (size_t i = 0; i < c->message_count; i++) {
        free(c->messages[i].role);
        free(c->messages[i].content);
    }
    
    // Reset message count
    c->message_count = 0;
    return CHATGPT_OK;
}

/*
 * Get the number of messages in the conversation
 * Usage: int count = chatgpt_get_message_count(client);
 * Returns: Number of messages, or 0 if client is NULL
 */
int chatgpt_get_message_count(const ChatGPTClient *c) {
    return c ? (int)c->message_count : 0;
}

/*
 * Remove the last message from the conversation
 * Useful for undoing the last message addition
 * Usage: chatgpt_pop_last_message(client); // Remove most recent message
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_pop_last_message(ChatGPTClient *c) {
    if (!c || c->message_count == 0) return CHATGPT_ERR_INVALID_ARG;
    
    // Get index of last message
    size_t i = c->message_count - 1;
    
    // Free the message content
    free(c->messages[i].role);
    free(c->messages[i].content);
    
    // Decrease count
    c->message_count--;
    return CHATGPT_OK;
}

/*
 * Remove a message at a specific index
 * All messages after the removed one will shift down by one position
 * Usage: chatgpt_remove_message_at(client, 2); // Remove the third message (0-indexed)
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_remove_message_at(ChatGPTClient *c, size_t idx) {
    if (!c || idx >= c->message_count) return CHATGPT_ERR_INVALID_ARG;
    
    // Free the message at the specified index
    free(c->messages[idx].role);
    free(c->messages[idx].content);
    
    // Shift all subsequent messages down
    for (size_t i = idx + 1; i < c->message_count; i++) {
        c->messages[i - 1] = c->messages[i];
    }
    
    // Decrease count
    c->message_count--;
    return CHATGPT_OK;
}

/*
 * Replace the content of the last user message
 * Searches backwards to find the most recent user message and replaces its content
 * Usage: chatgpt_replace_last_user(client, "Actually, let me ask something else...");
 * Returns: CHATGPT_OK on success, CHATGPT_ERR_STATE if no user message found
 */
int chatgpt_replace_last_user(ChatGPTClient *c, const char *txt) {
    if (!c || !txt) return CHATGPT_ERR_INVALID_ARG;
    
    // Search backwards for the last user message
    for (size_t i = c->message_count; i > 0; i--) {
        ChatGPTMessage *m = &c->messages[i - 1];
        if (m->role && strcmp(m->role, "user") == 0) {
            // Found user message, replace content
            char *d = dup_str(txt);
            if (!d) return CHATGPT_ERR_OOM;
            
            free(m->content);
            m->content = d;
            return CHATGPT_OK;
        }
    }
    
    // No user message found
    return CHATGPT_ERR_STATE;
}

/*
 * Append text to the last assistant message
 * Useful for building up responses incrementally
 * Usage: chatgpt_append_to_last_assistant(client, " Additional information...");
 * Returns: CHATGPT_OK on success, CHATGPT_ERR_STATE if no assistant message found
 */
int chatgpt_append_to_last_assistant(ChatGPTClient *c, const char *extra) {
    if (!c || !extra) return CHATGPT_ERR_INVALID_ARG;
    
    // Search backwards for the last assistant message
    for (size_t i = c->message_count; i > 0; i--) {
        ChatGPTMessage *m = &c->messages[i - 1];
        if (m->role && strcmp(m->role, "assistant") == 0) {
            // Found assistant message, append to content
            size_t a = m->content ? strlen(m->content) : 0;
            size_t b = strlen(extra);
            
            // Reallocate content to fit additional text
            char *p = (char*)realloc(m->content, a + b + 1);
            if (!p) return CHATGPT_ERR_OOM;
            
            // Append new text
            memcpy(p + a, extra, b + 1);
            m->content = p;
            return CHATGPT_OK;
        }
    }
    
    // No assistant message found
    return CHATGPT_ERR_STATE;
}

/*
 * Reset the client to a clean state
 * Clears messages, usage statistics, last reply, and errors
 * Keeps configuration settings (model, temperature, etc.)
 * Usage: chatgpt_reset(client); // Start completely fresh
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_reset(ChatGPTClient *c) {
    if (!c) return CHATGPT_ERR_INVALID_ARG;
    
    // Clear all messages
    chatgpt_clear_messages(c);
    
    // Reset usage statistics
    c->last_usage.prompt_tokens = 0;
    c->last_usage.completion_tokens = 0;
    c->last_usage.total_tokens = 0;
    
    // Clear last reply
    free(c->last_reply);
    c->last_reply = NULL;
    
    // Clear error state
    chatgpt_clear_error(c);
    
    return CHATGPT_OK;
}

/*
 * Create a JSON dump of all messages (alias for chatgpt_build_messages_json)
 * Usage: char *json = chatgpt_dump_messages(client); free(json); // Remember to free
 * Returns: JSON string or NULL on error
 */
char *chatgpt_dump_messages(ChatGPTClient *c) {
    return chatgpt_build_messages_json(c);
}

/*
 * Print all messages to a file stream for debugging
 * Useful for inspecting conversation state
 * Usage: chatgpt_print_messages(client, stdout); // Print to console
 *        chatgpt_print_messages(client, debug_file); // Print to file
 */
void chatgpt_print_messages(ChatGPTClient *c, FILE *out) {
    if (!c) return;
    if (!out) out = stdout;
    
    // Print each message with index, role, and content
    for (size_t i = 0; i < c->message_count; i++) {
        fprintf(out, "%zu %s: %s\n", 
                i, 
                c->messages[i].role ? c->messages[i].role : "?",
                c->messages[i].content ? c->messages[i].content : "");
    }
}

/*
 * Get the last reply from the AI (cached)
 * This is the complete response from the most recent API call
 * Usage: const char *reply = chatgpt_last_reply(client); // Do not free this pointer
 * Returns: Last reply string or NULL if no reply yet
 */
const char *chatgpt_last_reply(ChatGPTClient *c) {
    return c ? c->last_reply : NULL;
}

/*
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃           CONVERSATION PERSISTENCE            ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * Save the current conversation to a JSON file
 * Saves only the messages array, not configuration settings
 * Usage: chatgpt_save_conversation(client, "my_chat.json");
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_save_conversation(ChatGPTClient *c, const char *path) {
    if (!c || !path) return CHATGPT_ERR_INVALID_ARG;
    
    // Build JSON representation of messages
    char *j = chatgpt_build_messages_json(c);
    if (!j) return CHATGPT_ERR_OOM;
    
    // Open file for writing
    FILE *f = fopen(path, "w");
    if (!f) {
        free(j);
        return CHATGPT_ERR_HTTP;  // Reusing HTTP error for file I/O
    }
    
    // Write JSON to file
    fputs(j, f);
    fclose(f);
    free(j);
    
    return CHATGPT_OK;
}

/*
 * Load a conversation from a JSON file
 * Replaces current messages with those from the file
 * Usage: chatgpt_load_conversation(client, "my_chat.json");
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_load_conversation(ChatGPTClient *c, const char *path) {
    if (!c || !path) return CHATGPT_ERR_INVALID_ARG;
    
    // Open file for reading
    FILE *f = fopen(path, "r");
    if (!f) return CHATGPT_ERR_HTTP;
    
    // Get file size
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return CHATGPT_ERR_STATE;
    }
    fseek(f, 0, SEEK_SET);
    
    // Read entire file into buffer
    char *buf = (char*)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return CHATGPT_ERR_OOM;
    }
    
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    
    // Parse JSON
    cJSON *arr = cJSON_Parse(buf);
    free(buf);
    if (!arr) return CHATGPT_ERR_JSON_PARSE;
    
    // Verify it's an array
    if (!cJSON_IsArray(arr)) {
        cJSON_Delete(arr);
        return CHATGPT_ERR_JSON_PARSE;
    }
    
    // Clear existing messages
    chatgpt_clear_messages(c);
    
    // Add each message from JSON
    cJSON *it = NULL;
    cJSON_ArrayForEach(it, arr) {
        cJSON *r = cJSON_GetObjectItem(it, "role");
        cJSON *t = cJSON_GetObjectItem(it, "content");
        
        if (r && t && cJSON_IsString(r) && cJSON_IsString(t)) {
            chatgpt_add_message(c, r->valuestring, t->valuestring);
        }
    }
    
    cJSON_Delete(arr);
    return CHATGPT_OK;
}

/*
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃             API REQUEST BUILDING              ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * Build the JSON request body for OpenAI API
 * Creates a complete request with model, messages, and parameters
 * Internal function used by chat completion functions
 * Parameters:
 *   - stream: 1 for streaming mode, 0 for regular completion
 * Returns: JSON string or NULL on error (caller must free)
 */
static char *build_request_body(ChatGPTClient *c, int stream) {
    if (!c) return NULL;
    
    // Create root JSON object
    cJSON *root = cJSON_CreateObject();
    if (!root) return NULL;
    
    // Add model name
    cJSON_AddStringToObject(root, "model", c->model ? c->model : "gpt-4o-mini");
    
    // Create messages array
    cJSON *msgs = cJSON_CreateArray();
    if (!msgs) {
        cJSON_Delete(root);
        return NULL;
    }
    cJSON_AddItemToObject(root, "messages", msgs);
    
    // Add each message to the array
    for (size_t i = 0; i < c->message_count; i++) {
        cJSON *m = cJSON_CreateObject();
        if (!m) {
            cJSON_Delete(root);
            return NULL;
        }
        
        // Add role and content to message object
        cJSON_AddStringToObject(m, "role", c->messages[i].role);
        cJSON_AddStringToObject(m, "content", c->messages[i].content);
        cJSON_AddItemToArray(msgs, m);
    }
    
    // Add generation parameters
    cJSON_AddNumberToObject(root, "temperature", c->temperature);
    cJSON_AddNumberToObject(root, "top_p", c->top_p);
    
    // Add penalty parameters if they are not default (0.0)
    if (c->presence_penalty != 0.0) {
        cJSON_AddNumberToObject(root, "presence_penalty", c->presence_penalty);
    }
    if (c->frequency_penalty != 0.0) {
        cJSON_AddNumberToObject(root, "frequency_penalty", c->frequency_penalty);
    }
    
    // Add max_tokens if specified (0 means don't include it)
    if (c->max_tokens > 0) {
        cJSON_AddNumberToObject(root, "max_tokens", c->max_tokens);
    }
    
    // Add streaming flag if requested
    if (stream) {
        cJSON_AddBoolToObject(root, "stream", 1);
    }
    
    // Convert to string and cleanup
    char *out = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return out;
}

/* 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃            HTTP RESPONSE HANDLING             ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * Structure for accumulating HTTP response data
 * Used by curl to collect response data in chunks
 */
struct wb {
    char *d;    // Data buffer
    size_t n;   // Current size
};

/*
 * Curl write callback function
 * Called by curl for each chunk of response data received
 * Accumulates data in a growing buffer
 * Internal function used by HTTP requests
 */
static size_t write_cb(void *ptr, size_t sz, size_t nm, void *ud) {
    size_t need = sz * nm;  // Calculate total bytes received
    struct wb *w = (struct wb*)ud;
    
    // Reallocate buffer to fit new data
    char *p = realloc(w->d, w->n + need + 1);
    if (!p) return 0;  // Signal error to curl
    
    // Copy new data and update buffer
    w->d = p;
    memcpy(w->d + w->n, ptr, need);
    w->n += need;
    w->d[w->n] = '\0';  // Ensure null termination
    
    return need;  // Return bytes processed
}

/* 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃        CHAT COMPLETION (NON-STREAMING)        ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * Send a chat completion request and get the full response
 * This is the main function for getting AI responses
 * Usage: 
 *   chatgpt_add_user(client, "Hello!");
 *   char *response = chatgpt_chat_complete(client);
 *   printf("AI: %s\n", response);
 *   free(response);
 * Returns: Complete AI response as a new string (caller must free), or NULL on error
 */
char *chatgpt_chat_complete(ChatGPTClient *c) {
    char *body;                    // Request body JSON
    struct wb w = {0};             // Response buffer
    CURL *curl = NULL;             // Curl handle
    struct curl_slist *hdr = NULL; // HTTP headers
    CURLcode rc;                   // Curl result code
    char auth[512];                // Authorization header
    char url[512];                 // Complete API URL
    cJSON *root;                   // Parsed response JSON
    cJSON *err;                    // Error object from response
    cJSON *choices;                // Choices array from response
    cJSON *c0;                     // First choice
    cJSON *msg;                    // Message object
    cJSON *cont;                   // Content field
    cJSON *usage;                  // Usage statistics
    char *reply;                   // Final response text
    
    if (!c) return NULL;
    
    // Clear any previous error state
    chatgpt_clear_error(c);
    
    // Build request body JSON
    body = build_request_body(c, 0);  // 0 = non-streaming
    if (!body) {
        set_error(c, CHATGPT_ERR_OOM, "Failed to build request body");
        return NULL;
    }
    
    // Initialize curl
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (!curl) {
        free(body);
        set_error(c, CHATGPT_ERR_HTTP, "Failed to initialize curl");
        return NULL;
    }
    
    // Set up HTTP headers
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", c->api_key);
    hdr = curl_slist_append(hdr, "Content-Type: application/json");
    hdr = curl_slist_append(hdr, auth);
    
    // Build complete API URL
    snprintf(url, sizeof(url), "%s/v1/chat/completions", c->base_url);
    
    // Configure curl options
    curl_easy_setopt(curl, CURLOPT_URL, url);                    // Set URL
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdr);            // Set headers
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);           // Set POST data
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);    // Response handler
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&w);       // Response buffer
    
    // Perform the HTTP request
    rc = curl_easy_perform(curl);
    
    // Cleanup curl resources
    curl_slist_free_all(hdr);
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    free(body);
    
    // Check for HTTP errors
    if (rc != CURLE_OK) {
        free(w.d);
        set_error(c, CHATGPT_ERR_HTTP, curl_easy_strerror(rc));
        return NULL;
    }
    
    // Parse JSON response
    root = cJSON_Parse(w.d);
    if (!root) {
        free(w.d);
        set_error(c, CHATGPT_ERR_JSON_PARSE, "Failed to parse response JSON");
        return NULL;
    }
    
    // Check for API errors
    err = cJSON_GetObjectItem(root, "error");
    if (err) {
        cJSON *m = cJSON_GetObjectItem(err, "message");
        set_error(c, CHATGPT_ERR_API, 
                 (m && cJSON_IsString(m)) ? m->valuestring : "API returned error");
        cJSON_Delete(root);
        free(w.d);
        return NULL;
    }
    
    // Extract the response content
    choices = cJSON_GetObjectItem(root, "choices");
    if (!choices || !cJSON_IsArray(choices) || cJSON_GetArraySize(choices) == 0) {
        set_error(c, CHATGPT_ERR_JSON_PARSE, "No choices in response");
        cJSON_Delete(root);
        free(w.d);
        return NULL;
    }
    
    // Get first choice's message content
    c0 = cJSON_GetArrayItem(choices, 0);
    msg = cJSON_GetObjectItem(c0, "message");
    cont = msg ? cJSON_GetObjectItem(msg, "content") : NULL;
    
    if (!cont || !cJSON_IsString(cont)) {
        set_error(c, CHATGPT_ERR_JSON_PARSE, "No content in response message");
        cJSON_Delete(root);
        free(w.d);
        return NULL;
    }
    
    // Create response copy for return value
    reply = dup_str(cont->valuestring);
    
    // Cache the response in client
    free(c->last_reply);
    c->last_reply = dup_str(cont->valuestring);
    
    // Extract usage statistics if available
    usage = cJSON_GetObjectItem(root, "usage");
    if (usage) {
        cJSON *pt = cJSON_GetObjectItem(usage, "prompt_tokens");
        if (pt && cJSON_IsNumber(pt)) {
            c->last_usage.prompt_tokens = pt->valueint;
        }
        
        cJSON *ct = cJSON_GetObjectItem(usage, "completion_tokens");
        if (ct && cJSON_IsNumber(ct)) {
            c->last_usage.completion_tokens = ct->valueint;
        }
        
        cJSON *tt = cJSON_GetObjectItem(usage, "total_tokens");
        if (tt && cJSON_IsNumber(tt)) {
            c->last_usage.total_tokens = tt->valueint;
        }
    }
    
    // Cleanup and return
    cJSON_Delete(root);
    free(w.d);
    return reply;
}

/*
 * Get usage statistics from the last API call
 * Returns token counts for prompt, completion, and total usage
 * Usage: 
 *   ChatGPTUsage usage;
 *   if (chatgpt_get_last_usage(client, &usage) == CHATGPT_OK) {
 *     printf("Used %d tokens\n", usage.total_tokens);
 *   }
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_get_last_usage(ChatGPTClient *c, ChatGPTUsage *u) {
    if (!c || !u) return CHATGPT_ERR_INVALID_ARG;
    
    // Copy usage statistics
    *u = c->last_usage;
    return CHATGPT_OK;
}

/* 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃           STREAMING CHAT COMPLETION           ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ 
 */

/*
 * Context structure for streaming responses
 * Used to track callback function, user data, and accumulated response
 */
struct stream_ctx {
    chatgpt_stream_callback cb;  // User's callback function
    void *ud;                    // User data for callback
    char *acc;                   // Accumulated full response
    size_t len;                  // Length of accumulated response
};

/*
 * Curl write callback for streaming responses
 * Parses Server-Sent Events (SSE) format and extracts content deltas
 * Calls user callback for each content chunk received
 * Internal function used by streaming completion
 */
static size_t stream_cb(char *ptr, size_t sz, size_t nm, void *ud) {
    size_t tot = sz * nm;  // Total bytes received
    struct stream_ctx *ctx = (struct stream_ctx*)ud;
    
    // Process each line in the received data
    for (size_t i = 0; i < tot; ) {
        // Find end of current line
        size_t j = i;
        while (j < tot && ptr[j] != '\n') j++;
        
        size_t L = j - i;  // Line length
        
        // Check if this is a data line (SSE format: "data: ...")
        if (L > 5 && strncmp(&ptr[i], "data:", 5) == 0) {
            // Skip "data:" prefix and any spaces
            size_t k = i + 5;
            while (k < j && ptr[k] == ' ') k++;
            
            size_t pure = j - k;  // Length of actual data
            
            // Extract the data content
            char *line = (char*)malloc(pure + 1);
            if (line) {
                memcpy(line, &ptr[k], pure);
                line[pure] = '\0';
                
                // Check for end marker
                if (strcmp(line, "[DONE]") == 0) {
                    free(line);
                    return tot;  // Signal completion
                }
                
                // Parse JSON data
                cJSON *root = cJSON_Parse(line);
                if (root) {
                    // Navigate to content delta
                    cJSON *choices = cJSON_GetObjectItem(root, "choices");
                    if (choices && cJSON_IsArray(choices) && cJSON_GetArraySize(choices) > 0) {
                        cJSON *c0 = cJSON_GetArrayItem(choices, 0);
                        cJSON *delta = cJSON_GetObjectItem(c0, "delta");
                        if (delta) {
                            cJSON *content = cJSON_GetObjectItem(delta, "content");
                            if (content && cJSON_IsString(content)) {
                                // Call user callback with content delta
                                if (ctx->cb) {
                                    ctx->cb(content->valuestring, ctx->ud);
                                }
                                
                                // Accumulate content for full response
                                size_t add = strlen(content->valuestring);
                                char *np = (char*)realloc(ctx->acc, ctx->len + add + 1);
                                if (np) {
                                    ctx->acc = np;
                                    memcpy(ctx->acc + ctx->len, content->valuestring, add + 1);
                                    ctx->len += add;
                                }
                            }
                        }
                    }
                    cJSON_Delete(root);
                }
                free(line);
            }
        }
        
        // Move to next line
        i = (j < tot) ? j + 1 : j;
    }
    
    return tot;  // Return bytes processed
}

/*
 * Send a streaming chat completion request
 * Calls the provided callback function for each chunk of response text
 * Useful for real-time display of AI responses as they're generated
 * Usage:
 *   void my_callback(const char *delta, void *userdata) {
 *     printf("%s", delta);  // Print each chunk as it arrives
 *     fflush(stdout);
 *   }
 *   char *full_response;
 *   chatgpt_chat_complete_stream(client, my_callback, NULL, &full_response);
 *   free(full_response);
 * Parameters:
 *   - cb: Callback function called for each text chunk (can be NULL)
 *   - ud: User data passed to callback
 *   - full_out: Pointer to receive complete response (can be NULL)
 * Returns: CHATGPT_OK on success, error code on failure
 */
int chatgpt_chat_complete_stream(ChatGPTClient *c, chatgpt_stream_callback cb, 
                                void *ud, char **full_out) {
    char *body;                     // Request body JSON
    CURL *curl = NULL;             // Curl handle
    struct curl_slist *hdr = NULL; // HTTP headers
    char auth[512];                // Authorization header
    char url[512];                 // Complete API URL
    struct stream_ctx ctx;         // Streaming context
    CURLcode rc;                   // Curl result code
    
    if (!c) return CHATGPT_ERR_INVALID_ARG;
    
    // Clear any previous error state
    chatgpt_clear_error(c);
    
    // Build request body for streaming
    body = build_request_body(c, 1);  // 1 = streaming mode
    if (!body) {
        set_error(c, CHATGPT_ERR_OOM, "Failed to build request body");
        return CHATGPT_ERR_OOM;
    }
    
    // Initialize curl
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (!curl) {
        free(body);
        set_error(c, CHATGPT_ERR_HTTP, "Failed to initialize curl");
        return CHATGPT_ERR_HTTP;
    }
    
    // Set up HTTP headers
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", c->api_key);
    hdr = curl_slist_append(hdr, "Content-Type: application/json");
    hdr = curl_slist_append(hdr, auth);
    
    // Build complete API URL
    snprintf(url, sizeof(url), "%s/v1/chat/completions", c->base_url);
    
    // Initialize streaming context
    ctx.cb = cb;
    ctx.ud = ud;
    ctx.acc = NULL;
    ctx.len = 0;
    
    // Configure curl for streaming
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdr);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stream_cb);  // Streaming callback
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&ctx);
    
    // Perform the streaming request
    rc = curl_easy_perform(curl);
    
    // Cleanup curl resources
    curl_slist_free_all(hdr);
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    free(body);
    
    // Check for HTTP errors
    if (rc != CURLE_OK) {
        free(ctx.acc);
        set_error(c, CHATGPT_ERR_STREAM, curl_easy_strerror(rc));
        return CHATGPT_ERR_STREAM;
    }
    
    // Handle accumulated response
    if (full_out) {
        *full_out = ctx.acc;  // Give ownership to caller
        
        // Cache response in client
        free(c->last_reply);
        c->last_reply = dup_str(ctx.acc);
    } else {
        free(ctx.acc);  // Not needed by caller
    }
    
    return CHATGPT_OK;
}

/* 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃         LEGACY SINGLE PROMPT FUNCTION         ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * Simple one-shot query function
 * Creates a temporary client, sends a single user message, and returns the response
 * This is a convenience function for simple use cases
 * Usage: char *response = chatgpt_query("sk-your-key", "What is 2+2?");
 *        printf("Answer: %s\n", response);
 *        free(response);
 * Parameters:
 *   - api_key: OpenAI API key (can be NULL if global key is set)
 *   - prompt: User message to send
 * Returns: AI response string (caller must free), or NULL on error
 */
char *chatgpt_query(const char *api_key, const char *prompt) {
    
    // Create temporary client
    ChatGPTClient *c = chatgpt_client_new(api_key, NULL);
    char *r = NULL;
    
    if (!c) return NULL;
    
    // Add user message
    if (chatgpt_add_user(c, prompt) != 0) {
        chatgpt_client_free(c);
        return NULL;
    }
    
    // Get response
    r = chatgpt_chat_complete(c);
    
    // Cleanup and return
    chatgpt_client_free(c);
    return r;
}

/*
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                               ┃
┃              NEW FUNCTIONALITY                ┃
┃                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 */

/*
 * Get the last HTTP response code
 * Useful for identifying rate limits (429) and other HTTP-specific errors
 * Usage: long code = chatgpt_last_http_code(conversation);
 * Returns: HTTP response code from last API call
 */
long chatgpt_last_http_code(const ChatGPTConversation *c) {
    return c ? c->last_http_code : 0;
}

/*
 * Get a list of available models from the API
 * Returns a JSON string with available models (caller must free)
 * Usage: char *models = chatgpt_get_available_models(api_key); free(models);
 * Returns: JSON string or NULL on error
 */
char *chatgpt_get_available_models(const char *api_key) {
    if (!api_key) return NULL;
    
    struct wb w = {0};             // Response buffer
    CURL *curl = NULL;             // Curl handle
    struct curl_slist *hdr = NULL; // HTTP headers
    CURLcode rc;                   // Curl result code
    char auth[512];                // Authorization header
    char url[512];                 // Complete API URL
    
    // Initialize curl
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (!curl) return NULL;
    
    // Set up HTTP headers
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", api_key);
    hdr = curl_slist_append(hdr, auth);
    
    // Build complete API URL for models endpoint
    snprintf(url, sizeof(url), "https://api.openai.com/v1/models");
    
    // Configure curl options
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdr);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&w);
    
    // Perform the HTTP request
    rc = curl_easy_perform(curl);
    
    // Cleanup curl resources
    curl_slist_free_all(hdr);
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    
    // Check for HTTP errors
    if (rc != CURLE_OK) {
        free(w.d);
        return NULL;
    }
    
    return w.d; // Caller must free
}

/*
 * Check if a specific model is available
 * Returns 1 if available, 0 if not available, -1 on error
 * Usage: int available = chatgpt_is_model_available(api_key, "gpt-4");
 */
int chatgpt_is_model_available(const char *api_key, const char *model_name) {
    if (!api_key || !model_name) return -1;
    
    char *models_json = chatgpt_get_available_models(api_key);
    if (!models_json) return -1;
    
    // Simple substring search (could be improved with proper JSON parsing)
    int found = strstr(models_json, model_name) != NULL ? 1 : 0;
    
    free(models_json);
    return found;
}

/*
 * Generate an image using DALL-E
 * prompt: Description of the image to generate
 * size: Image size ("1024x1024", "512x512", "256x256")
 * Returns: URL to the generated image (caller must free) or NULL on error
 * Usage: char *url = chatgpt_generate_image(api_key, "A sunset", "512x512"); free(url);
 */
char *chatgpt_generate_image(const char *api_key, const char *prompt, const char *size) {
    if (!api_key || !prompt || !size) return NULL;
    
    struct wb w = {0};             // Response buffer
    CURL *curl = NULL;             // Curl handle
    struct curl_slist *hdr = NULL; // HTTP headers
    CURLcode rc;                   // Curl result code
    char auth[512];                // Authorization header
    char url[512];                 // Complete API URL
    char *body = NULL;             // Request body
    cJSON *root = NULL;            // JSON for request
    cJSON *response = NULL;        // JSON for response
    cJSON *data = NULL;            // Data array
    cJSON *first_item = NULL;      // First item in data
    cJSON *url_obj = NULL;         // URL object
    char *image_url = NULL;        // Final image URL
    
    // Create JSON request body
    root = cJSON_CreateObject();
    if (!root) return NULL;
    
    cJSON_AddStringToObject(root, "prompt", prompt);
    cJSON_AddNumberToObject(root, "n", 1);
    cJSON_AddStringToObject(root, "size", size);
    
    body = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (!body) return NULL;
    
    // Initialize curl
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (!curl) {
        free(body);
        return NULL;
    }
    
    // Set up HTTP headers
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", api_key);
    hdr = curl_slist_append(hdr, "Content-Type: application/json");
    hdr = curl_slist_append(hdr, auth);
    
    // Build complete API URL for image generation
    snprintf(url, sizeof(url), "https://api.openai.com/v1/images/generations");
    
    // Configure curl options
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdr);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&w);
    
    // Perform the HTTP request
    rc = curl_easy_perform(curl);
    
    // Cleanup curl resources
    curl_slist_free_all(hdr);
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    free(body);
    
    // Check for HTTP errors
    if (rc != CURLE_OK) {
        free(w.d);
        return NULL;
    }
    
    // Parse response JSON
    response = cJSON_Parse(w.d);
    free(w.d);
    if (!response) return NULL;
    
    // Extract image URL
    data = cJSON_GetObjectItem(response, "data");
    if (data && cJSON_IsArray(data) && cJSON_GetArraySize(data) > 0) {
        first_item = cJSON_GetArrayItem(data, 0);
        if (first_item) {
            url_obj = cJSON_GetObjectItem(first_item, "url");
            if (url_obj && cJSON_IsString(url_obj)) {
                image_url = dup_str(url_obj->valuestring);
            }
        }
    }
    
    cJSON_Delete(response);
    return image_url;
}

/*
 * Add a user message with an attached file (image or document)
 * file_path: Path to the file to attach
 * file_type: "image" for images, "document" for other files
 * content: Optional text content to accompany the file
 * Note: This is a placeholder implementation - full file upload requires base64 encoding
 */
int chatgpt_add_user_with_file(ChatGPTConversation *c, const char *content, 
                               const char *file_path, const char *file_type) {
    if (!c || !file_path || !file_type) return CHATGPT_ERR_INVALID_ARG;
    
    // For now, just add a text message indicating file attachment
    // Full implementation would require base64 encoding of the file
    char *message = (char*)malloc(strlen(content ? content : "") + strlen(file_path) + 100);
    if (!message) return CHATGPT_ERR_OOM;
    
    snprintf(message, strlen(content ? content : "") + strlen(file_path) + 99,
             "%s [File attached: %s (%s)]", 
             content ? content : "File attachment", file_path, file_type);
    
    int result = chatgpt_add_user(c, message);
    free(message);
    
    return result;
}