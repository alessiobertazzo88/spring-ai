/*
 * Copyright 2023 - 2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.springframework.ai.vertexai.anthropic;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.model.*;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.model.function.FunctionCallback;
import org.springframework.ai.model.function.FunctionCallbackContext;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.ai.vertexai.anthropic.api.VertexAIAnthropicApi;
import org.springframework.ai.vertexai.anthropic.model.*;
import org.springframework.http.ResponseEntity;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Java {@link ChatModel} and {@link StreamingChatModel} for the VertexAI Anthropic chat
 * generative.
 *
 * @author Alessio Bertazzo
 * @since 1.0.0
 */
public class VertexAIAnthropicChatModel extends AbstractToolCallSupport implements ChatModel, StreamingChatModel {

	private final VertexAIAnthropicApi anthropicApi;

	private final VertexAIAnthropicChatOptions defaultOptions;

	/**
	 * The retry template used to retry the VertexAI Anthropic API calls.
	 */
	public final RetryTemplate retryTemplate;

	private static final String DEFAULT_ANTHROPIC_VERSION = "vertex-2023-10-16";

	public VertexAIAnthropicChatModel(VertexAIAnthropicApi anthropicApi) {
		this(anthropicApi,
				VertexAIAnthropicChatOptions.builder()
					.withTemperature(0.8f)
					.withMaxTokens(500)
					.withTopK(10)
					.withAnthropicVersion(DEFAULT_ANTHROPIC_VERSION)
					.withModel(ChatModels.CLAUDE_3_5_SONNET.getValue())
					.build());
	}

	public VertexAIAnthropicChatModel(VertexAIAnthropicApi anthropicApi, VertexAIAnthropicChatOptions options) {
		this(anthropicApi, options, RetryUtils.DEFAULT_RETRY_TEMPLATE);
	}

	public VertexAIAnthropicChatModel(VertexAIAnthropicApi anthropicApi, VertexAIAnthropicChatOptions options,
			RetryTemplate retryTemplate) {
		this(anthropicApi, options, retryTemplate, null);
	}

	public VertexAIAnthropicChatModel(VertexAIAnthropicApi anthropicApi, VertexAIAnthropicChatOptions options,
			RetryTemplate retryTemplate, FunctionCallbackContext functionCallbackContext) {
		this(anthropicApi, options, retryTemplate, functionCallbackContext, List.of());
	}

	public VertexAIAnthropicChatModel(VertexAIAnthropicApi anthropicApi, VertexAIAnthropicChatOptions options,
			RetryTemplate retryTemplate, FunctionCallbackContext functionCallbackContext,
			List<FunctionCallback> toolFunctionCallbacks) {

		super(functionCallbackContext, options, toolFunctionCallbacks);

		Assert.notNull(anthropicApi, "VertexAIAnthropicApi must not be null");
		Assert.notNull(options, "VertexAiAnthropicChatOptions must not be null");

		this.anthropicApi = anthropicApi;
		this.defaultOptions = options;
		this.retryTemplate = retryTemplate;
	}

	@Override
	public ChatResponse call(Prompt prompt) {
		ChatCompletionRequest request = createRequest(prompt, false);

		ResponseEntity<ChatCompletionResponse> completionEntity = this.retryTemplate
			.execute(ctx -> this.anthropicApi.chatCompletion(request, defaultOptions.getModel()));

		ChatResponse chatResponse = toChatResponse(completionEntity.getBody());

		if (this.isToolCall(chatResponse, Set.of("tool_use"))) {
			var toolCallConversation = handleToolCalls(prompt, chatResponse);
			return this.call(new Prompt(toolCallConversation, prompt.getOptions()));
		}

		return chatResponse;
	}

	@Override
	public Flux<ChatResponse> stream(Prompt prompt) {
		ChatCompletionRequest request = createRequest(prompt, true);

		Flux<ChatCompletionResponse> response = this.retryTemplate
			.execute(ctx -> this.anthropicApi.chatCompletionStream(request, defaultOptions.getModel()));

		return response.switchMap(chatCompletionResponse -> {

			ChatResponse chatResponse = toChatResponse(chatCompletionResponse);

			if (this.isToolCall(chatResponse, Set.of("tool_use"))) {
				var toolCallConversation = handleToolCalls(prompt, chatResponse);
				return this.stream(new Prompt(toolCallConversation, prompt.getOptions()));
			}

			return Mono.just(chatResponse);
		});
	}

	ChatCompletionRequest createRequest(Prompt prompt, boolean stream) {

		Set<String> functionsForThisRequest = new HashSet<>();

		List<AnthropicMessage> userMessages = prompt.getInstructions()
			.stream()
			.filter(message -> message.getMessageType() != MessageType.SYSTEM)
			.map(message -> {
				if (message.getMessageType() == MessageType.USER) {
					List<ContentBlock> contents = new ArrayList<>(List.of(new ContentBlock(message.getContent())));
					if (message instanceof UserMessage userMessage) {
						if (!CollectionUtils.isEmpty(userMessage.getMedia())) {
							List<ContentBlock> mediaContent = userMessage.getMedia()
								.stream()
								.map(media -> new ContentBlock(media.getMimeType().toString(),
										this.fromMediaData(media.getData())))
								.toList();
							contents.addAll(mediaContent);
						}
					}
					return new AnthropicMessage(contents, Role.valueOf(message.getMessageType().name()));
				}
				else if (message.getMessageType() == MessageType.ASSISTANT) {
					AssistantMessage assistantMessage = (AssistantMessage) message;
					List<ContentBlock> contentBlocks = new ArrayList<>();
					if (StringUtils.hasText(message.getContent())) {
						contentBlocks.add(new ContentBlock(message.getContent()));
					}
					if (!CollectionUtils.isEmpty(assistantMessage.getToolCalls())) {
						for (AssistantMessage.ToolCall toolCall : assistantMessage.getToolCalls()) {
							contentBlocks.add(new ContentBlock(ContentBlock.Type.TOOL_USE, toolCall.id(),
									toolCall.name(), ModelOptionsUtils.jsonToMap(toolCall.arguments())));
						}
					}
					return new AnthropicMessage(contentBlocks, Role.ASSISTANT);
				}
				else if (message.getMessageType() == MessageType.TOOL) {
					List<ContentBlock> toolResponses = ((ToolResponseMessage) message).getResponses()
						.stream()
						.map(toolResponse -> new ContentBlock(ContentBlock.Type.TOOL_RESULT, toolResponse.id(),
								toolResponse.responseData()))
						.toList();
					return new AnthropicMessage(toolResponses, Role.USER);
				}
				else {
					throw new IllegalArgumentException("Unsupported message type: " + message.getMessageType());
				}
			})
			.toList();

		String systemPrompt = prompt.getInstructions()
			.stream()
			.filter(m -> m.getMessageType() == MessageType.SYSTEM)
			.map(m -> m.getContent())
			.collect(Collectors.joining(System.lineSeparator()));

		ChatCompletionRequest request = new ChatCompletionRequest(this.defaultOptions.getModel(), userMessages,
				systemPrompt, this.defaultOptions.getMaxTokens(), this.defaultOptions.getTemperature(), stream);

		if (prompt.getOptions() != null) {
			VertexAIAnthropicChatOptions updatedRuntimeOptions = ModelOptionsUtils.copyToTarget(prompt.getOptions(),
					ChatOptions.class, VertexAIAnthropicChatOptions.class);

			functionsForThisRequest.addAll(this.runtimeFunctionCallbackConfigurations(updatedRuntimeOptions));

			request = ModelOptionsUtils.merge(updatedRuntimeOptions, request, ChatCompletionRequest.class);
		}

		if (!CollectionUtils.isEmpty(this.defaultOptions.getFunctions())) {
			functionsForThisRequest.addAll(this.defaultOptions.getFunctions());
		}

		request = ModelOptionsUtils.merge(request, this.defaultOptions, ChatCompletionRequest.class);

		if (!CollectionUtils.isEmpty(functionsForThisRequest)) {

			List<Tool> tools = getFunctionTools(functionsForThisRequest);

			request = ChatCompletionRequest.from(request).withTools(tools).build();
		}

		return request;
	}

	private String fromMediaData(Object mediaData) {
		if (mediaData instanceof byte[] bytes) {
			return Base64.getEncoder().encodeToString(bytes);
		}
		else if (mediaData instanceof String text) {
			return text;
		}
		else {
			throw new IllegalArgumentException("Unsupported media data type: " + mediaData.getClass().getSimpleName());
		}
	}

	private List<Tool> getFunctionTools(Set<String> functionNames) {
		return this.resolveFunctionCallbacks(functionNames).stream().map(functionCallback -> {
			var description = functionCallback.getDescription();
			var name = functionCallback.getName();
			String inputSchema = functionCallback.getInputTypeSchema();
			return new Tool(name, description, ModelOptionsUtils.jsonToMap(inputSchema));
		}).toList();
	}

	private ChatResponse toChatResponse(ChatCompletionResponse chatCompletion) {

		if (chatCompletion == null) {
			return new ChatResponse(List.of());
		}

		List<Generation> generations = chatCompletion.content()
			.stream()
			.filter(content -> content.type() != ContentBlock.Type.TOOL_USE)
			.map(content -> {
				return new Generation(new AssistantMessage(content.text(), Map.of()),
						ChatGenerationMetadata.from(chatCompletion.stopReason(), null));
			})
			.toList();

		List<Generation> allGenerations = new ArrayList<>(generations);

		List<ContentBlock> toolToUseList = chatCompletion.content()
			.stream()
			.filter(c -> c.type() == ContentBlock.Type.TOOL_USE)
			.toList();

		if (!CollectionUtils.isEmpty(toolToUseList)) {
			List<AssistantMessage.ToolCall> toolCalls = new ArrayList<>();

			for (ContentBlock toolToUse : toolToUseList) {

				var functionCallId = toolToUse.id();
				var functionName = toolToUse.name();
				var functionArguments = ModelOptionsUtils.toJsonString(toolToUse.input());

				toolCalls
					.add(new AssistantMessage.ToolCall(functionCallId, "function", functionName, functionArguments));
			}

			AssistantMessage assistantMessage = new AssistantMessage("", Map.of(), toolCalls);
			Generation toolCallGeneration = new Generation(assistantMessage,
					ChatGenerationMetadata.from(chatCompletion.stopReason(), null));
			allGenerations.add(toolCallGeneration);
		}

		return new ChatResponse(allGenerations, this.from(chatCompletion));
	}

	private ChatResponseMetadata from(ChatCompletionResponse result) {
		Assert.notNull(result, "Anthropic ChatCompletionResult must not be null");
		AnthropicUsage usage = AnthropicUsage.from(result.usage());
		return ChatResponseMetadata.builder()
			.withId(result.id())
			.withModel(result.model())
			.withUsage(usage)
			.withKeyValue("stop-reason", result.stopReason())
			.withKeyValue("stop-sequence", result.stopSequence())
			.withKeyValue("type", result.type())
			.build();
	}

	@Override
	public ChatOptions getDefaultOptions() {
		return VertexAIAnthropicChatOptions.fromOptions(this.defaultOptions);
	}

}
