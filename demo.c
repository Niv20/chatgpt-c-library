/*
 * ChatGPT C Library - New Conversation-Based Demo
 * This demo shows the new conversation-based interface with streaming and context management
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chatgpt.h"

// Callback function for streaming responses
static void stream_callback(const char *delta, void *user_data) {
    (void)user_data; // Unused parameter
    printf("%s", delta);
    fflush(stdout);
}

int main(void) {
    printf("ChatGPT C Library - New Conversation Demo\n");
    printf("=========================================\n\n");

    // IMPORTANT: Replace with your actual OpenAI API key
    const char *api_key = "PUT_YOUR_OPENAI_API_KEY_HERE";
    
    if (strcmp(api_key, "PUT_YOUR_OPENAI_API_KEY_HERE") == 0) {
        printf("⚠️  Please set your OpenAI API key in the source code!\n");
        printf("   Replace 'PUT_YOUR_OPENAI_API_KEY_HERE' with your real key.\n\n");
        return 1;
    }

    // ========== EXAMPLE 1: Basic Conversation with New Interface ==========
    printf("1. Basic Conversation (New Interface)\n");
    printf("=====================================\n");
    
    // Create a new conversation (replaces client)
    ChatGPTConversation *conv = chatgpt_conversation_new(api_key, "gpt-4o-mini");
    if (!conv) {
        printf("❌ Failed to create conversation\n");
        return 1;
    }

    // Configure conversation settings
    chatgpt_set_temperature(conv, 0.8);
    chatgpt_set_streaming(conv, 1);  // Enable streaming (default)
    chatgpt_set_context_messages(conv, 3);  // Only send last 3 messages
    
    printf("📋 Configuration:\n");
    printf("  - Streaming: Enabled\n");
    printf("  - Context messages: 3\n");
    printf("  - Temperature: 0.8\n\n");

    // Set up the AI's behavior
    chatgpt_add_system(conv, "You are a helpful coding assistant. Be concise but informative.");
    
    // First question
    chatgpt_add_user(conv, "What is the difference between malloc and calloc in C?");
    printf("User: What is the difference between malloc and calloc in C?\n");
    printf("AI (streaming): ");
    
    char *response1 = NULL;
    int result = chatgpt_chat_complete_stream(conv, stream_callback, NULL, &response1);
    if (result == CHATGPT_OK) {
        printf("\n✅ Response received successfully\n\n");
        free(response1);
    } else {
        printf("\n❌ Error: %s\n\n", chatgpt_last_error(conv));
    }

    // Follow-up question
    chatgpt_add_user(conv, "Can you show a simple example?");
    printf("User: Can you show a simple example?\n");
    printf("AI (streaming): ");
    
    char *response2 = NULL;
    result = chatgpt_chat_complete_stream(conv, stream_callback, NULL, &response2);
    if (result == CHATGPT_OK) {
        printf("\n✅ Response received successfully\n\n");
        free(response2);
    } else {
        printf("\n❌ Error: %s\n\n", chatgpt_last_error(conv));
    }

    // ========== EXAMPLE 2: Context Management ==========
    printf("2. Context Management Demo\n");
    printf("==========================\n");
    
    printf("Current messages in conversation: %d\n", chatgpt_get_message_count(conv));
    
    // Add several more messages
    chatgpt_add_user(conv, "Message 1");
    chatgpt_add_assistant(conv, "Response 1");
    chatgpt_add_user(conv, "Message 2"); 
    chatgpt_add_assistant(conv, "Response 2");
    chatgpt_add_user(conv, "Message 3");
    
    printf("After adding more messages: %d\n", chatgpt_get_message_count(conv));
    printf("With context_messages=3, only the last 3 messages will be sent to API\n\n");

    // ========== EXAMPLE 3: Configuration Copying ==========
    printf("3. Configuration Copying\n");
    printf("========================\n");
    
    // Create a second conversation
    ChatGPTConversation *conv2 = chatgpt_conversation_new(api_key, "gpt-3.5-turbo");
    if (conv2) {
        printf("Created second conversation with gpt-3.5-turbo\n");
        
        // Copy settings from first conversation
        if (chatgpt_conversation_copy_settings(conv2, conv) == CHATGPT_OK) {
            printf("✅ Settings copied successfully\n");
            printf("Second conversation now has same settings but different messages\n\n");
        } else {
            printf("❌ Failed to copy settings\n\n");
        }
        
        chatgpt_conversation_free(conv2);
    }

    // ========== EXAMPLE 4: Non-Streaming Mode ==========
    printf("4. Non-Streaming Mode\n");
    printf("=====================\n");
    
    // Disable streaming for complete response
    chatgpt_set_streaming(conv, 0);
    printf("Streaming disabled - waiting for complete response...\n");
    
    chatgpt_add_user(conv, "What are the benefits of the C programming language?");
    printf("User: What are the benefits of the C programming language?\n");
    
    char *complete_response = chatgpt_chat_complete(conv);
    if (complete_response) {
        printf("AI (complete): %s\n\n", complete_response);
        free(complete_response);
    } else {
        printf("❌ Error: %s\n\n", chatgpt_last_error(conv));
    }

    // ========== EXAMPLE 5: Error Handling and HTTP Codes ==========
    printf("5. Error Handling\n");
    printf("=================\n");
    
    // Try an invalid model to demonstrate error handling
    ChatGPTConversation *conv3 = chatgpt_conversation_new(api_key, "invalid-model-name");
    if (conv3) {
        chatgpt_add_user(conv3, "This should fail");
        char *response = chatgpt_chat_complete(conv3);
        
        if (!response) {
            printf("Expected error occurred:\n");
            printf("Error code: %d\n", chatgpt_last_code(conv3));
            printf("Error message: %s\n", chatgpt_last_error(conv3));
            printf("HTTP code: %ld\n\n", chatgpt_last_http_code(conv3));
        }
        
        chatgpt_conversation_free(conv3);
    }

    // ========== EXAMPLE 6: Available Models Check ==========
    printf("6. Available Models\n");
    printf("===================\n");
    
    // Check if a model is available
    int model_available = chatgpt_is_model_available(api_key, "gpt-4o-mini");
    if (model_available == 1) {
        printf("✅ gpt-4o-mini is available\n");
    } else if (model_available == 0) {
        printf("❌ gpt-4o-mini is not available\n");
    } else {
        printf("❓ Error checking model availability\n");
    }
    
    // Get list of available models
    char *models_json = chatgpt_get_available_models(api_key);
    if (models_json) {
        printf("📋 Available models (JSON): %s\n\n", models_json);
        free(models_json);
    } else {
        printf("❌ Failed to get available models\n\n");
    }

    // ========== EXAMPLE 7: Image Generation ==========
    printf("7. Image Generation\n");
    printf("===================\n");
    
    char *image_url = chatgpt_generate_image(api_key, 
        "A beautiful sunset over mountains with a lake in the foreground", 
        "512x512");
    
    if (image_url) {
        printf("🎨 Generated image URL: %s\n\n", image_url);
        free(image_url);
    } else {
        printf("❌ Failed to generate image\n\n");
    }

    // ========== EXAMPLE 8: File Attachment (Conceptual) ==========
    printf("8. File Attachment (Future Feature)\n");
    printf("====================================\n");
    
    // This would be the syntax for file attachments (implementation needed)
    printf("Future syntax for file attachments:\n");
    printf("chatgpt_add_user_with_file(conv, \"Analyze this image\", \"image.jpg\", \"image\");\n\n");

    // ========== CLEANUP ==========
    printf("==================================================\n");
    printf("Demo completed! Cleaning up...\n");
    
    chatgpt_conversation_free(conv);
    printf("✅ Conversation freed\n");
    
    printf("\nNew features demonstrated:\n");
    printf("- Conversation-based interface (replaces Client)\n");
    printf("- Streaming enabled by default\n");
    printf("- Context message management\n");
    printf("- Configuration copying between conversations\n");
    printf("- Enhanced error handling with HTTP codes\n");
    printf("- Model availability checking\n");
    printf("- Image generation capability\n");

    return 0;
}
