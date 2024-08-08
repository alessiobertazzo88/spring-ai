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

import com.fasterxml.jackson.annotation.JsonInclude;
import com.google.cloud.vertexai.VertexAI;
import com.google.cloud.vertexai.api.*;
import com.google.cloud.vertexai.generativeai.GenerativeModel;
import com.google.cloud.vertexai.generativeai.PartMaker;
import com.google.cloud.vertexai.generativeai.ResponseStream;
import com.google.protobuf.Struct;
import com.google.protobuf.util.JsonFormat;
import org.springframework.ai.chat.messages.*;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.model.*;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.ChatModelDescription;
import org.springframework.ai.model.Media;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.model.function.FunctionCallback;
import org.springframework.ai.model.function.FunctionCallbackContext;
import org.springframework.ai.vertexai.anthropic.metadata.VertexAiUsage;
import org.springframework.lang.NonNull;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;
import reactor.core.publisher.Flux;

import java.util.*;

/**
 * Java {@link ChatModel} and {@link StreamingChatModel} for the VertexAI Anthropic chat
 * generative.
 *
 * @author Alessio Bertazzo
 * @since 1.0.0
 */
public class VertexAIAnthropicChatModel extends AbstractToolCallSupport implements ChatModel, StreamingChatModel {

	private final VertexAI vertexAI;

	private final VertexAIAnthropicChatOptions defaultOptions;

	private final GenerationConfig generationConfig;

	public enum AnthropicMessageType {

		USER("user"),

		MODEL("model");

		AnthropicMessageType(String value) {
			this.value = value;
		}

		public final String value;

		public String getValue() {
			return this.value;
		}

	}

	public enum ChatModel implements ChatModelDescription {

		ANTHROPIC("vertex-2023-10-16");

		ChatModel(String value) {
			this.value = value;
		}

		public final String value;

		public String getValue() {
			return this.value;
		}

		@Override
		public String getName() {
			return this.value;
		}

	}

	public VertexAIAnthropicChatModel(VertexAI vertexAI) {
		this(vertexAI,
				VertexAIAnthropicChatOptions.builder()
					.withTemperature(0.8f)
					.withMaxOutputTokens(500)
					.withTopK(10)
					.withAnthropicVersion(ChatModel.ANTHROPIC.value)
					.build());
	}

	public VertexAIAnthropicChatModel(VertexAI vertexAI, VertexAIAnthropicChatOptions options) {
		this(vertexAI, options, null);
	}

	public VertexAIAnthropicChatModel(VertexAI vertexAI, VertexAIAnthropicChatOptions options,
			FunctionCallbackContext functionCallbackContext) {
		this(vertexAI, options, functionCallbackContext, List.of());
	}

	public VertexAIAnthropicChatModel(VertexAI vertexAI, VertexAIAnthropicChatOptions options,
			FunctionCallbackContext functionCallbackContext, List<FunctionCallback> toolFunctionCallbacks) {

		super(functionCallbackContext, options, toolFunctionCallbacks);

		Assert.notNull(vertexAI, "VertexAI must not be null");
		Assert.notNull(options, "VertexAiGeminiChatOptions must not be null");

		this.vertexAI = vertexAI;
		this.defaultOptions = options;
		this.generationConfig = toGenerationConfig(options);
	}

	@Override
	public ChatResponse call(Prompt prompt) {
		AnthropicRequest anthropicRequest = createAnthropicRequest(prompt);

		GenerateContentResponse response = this.getContentResponse(anthropicRequest);

		List<Generation> generations = response.getCandidatesList()
			.stream()
			.map(this::responseCandiateToGeneration)
			.flatMap(List::stream)
			.toList();

		ChatResponse chatResponse = new ChatResponse(generations, toChatResponseMetadata(response));

		if (isToolCall(chatResponse, Set.of(Candidate.FinishReason.STOP.name()))) {
			var toolCallConversation = handleToolCalls(prompt, chatResponse);
			// Recursively call the call method with the tool call message
			// conversation that contains the call responses.
			return this.call(new Prompt(toolCallConversation, prompt.getOptions()));
		}

		return chatResponse;
	}

	@Override
	public Flux<ChatResponse> stream(Prompt prompt) {
		try {
			AnthropicRequest anthropicRequest = createAnthropicRequest(prompt);

			ResponseStream<GenerateContentResponse> responseStream = anthropicRequest.model
				.generateContentStream(anthropicRequest.contents);

			return Flux.fromStream(responseStream.stream()).switchMap(response -> {

				List<Generation> generations = response.getCandidatesList()
					.stream()
					.map(this::responseCandiateToGeneration)
					.flatMap(List::stream)
					.toList();

				ChatResponse chatResponse = new ChatResponse(generations, toChatResponseMetadata(response));

				if (isToolCall(chatResponse, Set.of(Candidate.FinishReason.STOP.name(),
						Candidate.FinishReason.FINISH_REASON_UNSPECIFIED.name()))) {
					var toolCallConversation = handleToolCalls(prompt, chatResponse);
					// Recursively call the stream method with the tool call message
					// conversation that contains the call responses.
					return this.stream(new Prompt(toolCallConversation, prompt.getOptions()));
				}

				return Flux.just(chatResponse);
			});
		}
		catch (Exception e) {
			throw new RuntimeException("Failed to generate content", e);
		}
	}

	protected List<Generation> responseCandiateToGeneration(Candidate candidate) {

		// TODO - The candidateIndex (e.g. choice must be asigned to the generation).
		int candidateIndex = candidate.getIndex();
		Candidate.FinishReason candidateFinishReasonn = candidate.getFinishReason();

		Map<String, Object> messageMetadata = Map.of("candidateIndex", candidateIndex, "finishReason",
				candidateFinishReasonn);

		ChatGenerationMetadata chatGenerationMetadata = ChatGenerationMetadata.from(candidateFinishReasonn.name(),
				null);

		boolean isFunctinCall = candidate.getContent().getPartsList().stream().allMatch(Part::hasFunctionCall);

		if (isFunctinCall) {
			List<AssistantMessage.ToolCall> assistantToolCalls = candidate.getContent()
				.getPartsList()
				.stream()
				.filter(part -> part.hasFunctionCall())
				.map(part -> {
					FunctionCall functionCall = part.getFunctionCall();
					var functionName = functionCall.getName();
					String functionArguments = structToJson(functionCall.getArgs());
					return new AssistantMessage.ToolCall("", "function", functionName, functionArguments);
				})
				.toList();

			AssistantMessage assistantMessage = new AssistantMessage("", messageMetadata, assistantToolCalls);

			return List.of(new Generation(assistantMessage, chatGenerationMetadata));
		}
		else {
			List<Generation> generations = candidate.getContent()
				.getPartsList()
				.stream()
				.map(part -> new AssistantMessage(part.getText(), messageMetadata))
				.map(assistantMessage -> new Generation(assistantMessage, chatGenerationMetadata))
				.toList();

			return generations;
		}
	}

	private ChatResponseMetadata toChatResponseMetadata(GenerateContentResponse response) {
		return ChatResponseMetadata.builder().withUsage(new VertexAiUsage(response.getUsageMetadata())).build();
	}

	@JsonInclude(JsonInclude.Include.NON_NULL)
	public record AnthropicRequest(List<Content> contents, GenerativeModel model) {
	}

	AnthropicRequest createAnthropicRequest(Prompt prompt) {

		Set<String> functionsForThisRequest = new HashSet<>();

		GenerationConfig generationConfig = this.generationConfig;

		var generativeModelBuilder = new GenerativeModel.Builder()
			.setModelName(this.defaultOptions.getAnthropicVersion())
			.setVertexAi(this.vertexAI);

		VertexAIAnthropicChatOptions updatedRuntimeOptions = VertexAIAnthropicChatOptions.builder().build();

		if (prompt.getOptions() != null) {
			updatedRuntimeOptions = ModelOptionsUtils.copyToTarget(prompt.getOptions(), ChatOptions.class,
					VertexAIAnthropicChatOptions.class);

			functionsForThisRequest.addAll(runtimeFunctionCallbackConfigurations(updatedRuntimeOptions));
		}

		if (!CollectionUtils.isEmpty(this.defaultOptions.getFunctions())) {
			functionsForThisRequest.addAll(this.defaultOptions.getFunctions());
		}

		updatedRuntimeOptions = ModelOptionsUtils.merge(updatedRuntimeOptions, this.defaultOptions,
				VertexAIAnthropicChatOptions.class);

		if (updatedRuntimeOptions != null) {

			if (StringUtils.hasText(updatedRuntimeOptions.getAnthropicVersion())
					&& !updatedRuntimeOptions.getAnthropicVersion().equals(this.defaultOptions.getAnthropicVersion())) {
				// Override model name
				generativeModelBuilder.setModelName(updatedRuntimeOptions.getAnthropicVersion());
			}

			generationConfig = toGenerationConfig(updatedRuntimeOptions);
		}

		// Add the enabled functions definitions to the request's tools parameter.
		if (!CollectionUtils.isEmpty(functionsForThisRequest)) {
			List<Tool> tools = this.getFunctionTools(functionsForThisRequest);
			generativeModelBuilder.setTools(tools);
		}

		generativeModelBuilder.setGenerationConfig(generationConfig);

		GenerativeModel generativeModel = generativeModelBuilder.build();

		List<Content> contents = toAnthropicContent(
				prompt.getInstructions().stream().filter(m -> m.getMessageType() == MessageType.SYSTEM).toList());

		if (!CollectionUtils.isEmpty(contents)) {
			Assert.isTrue(contents.size() <= 1, "Only one system message is allowed in the prompt");
			generativeModel = generativeModel.withSystemInstruction(contents.get(0));
		}

		return new AnthropicRequest(toAnthropicContent(
				prompt.getInstructions().stream().filter(m -> m.getMessageType() != MessageType.SYSTEM).toList()),
				generativeModel);
	}

	private GenerationConfig toGenerationConfig(VertexAIAnthropicChatOptions options) {

		GenerationConfig.Builder generationConfigBuilder = GenerationConfig.newBuilder();

		if (options.getTemperature() != null) {
			generationConfigBuilder.setTemperature(options.getTemperature());
		}
		if (options.getMaxOutputTokens() != null) {
			generationConfigBuilder.setMaxOutputTokens(options.getMaxOutputTokens());
		}
		if (options.getTopK() != null) {
			generationConfigBuilder.setTopK(options.getTopK());
		}
		if (options.getTopP() != null) {
			generationConfigBuilder.setTopP(options.getTopP());
		}
		if (options.getStopSequences() != null) {
			generationConfigBuilder.addAllStopSequences(options.getStopSequences());
		}

		return generationConfigBuilder.build();
	}

	private List<Content> toAnthropicContent(List<Message> instructions) {

		List<Content> contents = instructions.stream()
			.map(message -> Content.newBuilder()
				.setRole(toGeminiMessageType(message.getMessageType()).getValue())
				.addAllParts(messageToGeminiParts(message))
				.build())
			.toList();

		return contents;
	}

	private static AnthropicMessageType toGeminiMessageType(@NonNull MessageType type) {
		Assert.notNull(type, "Message type must not be null");

		switch (type) {
			case SYSTEM:
			case USER:
			case TOOL:
				return AnthropicMessageType.USER;
			case ASSISTANT:
				return AnthropicMessageType.MODEL;
			default:
				throw new IllegalArgumentException("Unsupported message type: " + type);
		}
	}

	static List<Part> messageToGeminiParts(Message message) {
		if (message instanceof SystemMessage systemMessage) {
			List<Part> parts = new ArrayList<>();

			if (systemMessage.getContent() != null) {
				parts.add(Part.newBuilder().setText(systemMessage.getContent()).build());
			}

			return parts;
		}
		else if (message instanceof UserMessage userMessage) {
			List<Part> parts = new ArrayList<>();
			if (userMessage.getContent() != null) {
				parts.add(Part.newBuilder().setText(userMessage.getContent()).build());
			}

			parts.addAll(mediaToParts(userMessage.getMedia()));

			return parts;
		}
		else if (message instanceof AssistantMessage assistantMessage) {
			List<Part> parts = new ArrayList<>();
			if (StringUtils.hasText(assistantMessage.getContent())) {
				List.of(Part.newBuilder().setText(assistantMessage.getContent()).build());
			}
			if (!CollectionUtils.isEmpty(assistantMessage.getToolCalls())) {
				parts.addAll(assistantMessage.getToolCalls()
					.stream()
					.map(toolCall -> Part.newBuilder()
						.setFunctionCall(FunctionCall.newBuilder()
							.setName(toolCall.name())
							.setArgs(jsonToStruct(toolCall.arguments()))
							.build())
						.build())
					.toList());
			}
			return parts;
		}
		else if (message instanceof ToolResponseMessage toolResponseMessage) {

			return toolResponseMessage.getResponses()
				.stream()
				.map(response -> Part.newBuilder()
					.setFunctionResponse(FunctionResponse.newBuilder()
						.setName(response.name())
						.setResponse(jsonToStruct(response.responseData()))
						.build())
					.build())
				.toList();
		}
		else {
			throw new IllegalArgumentException("Gemini doesn't support message type: " + message.getClass());
		}
	}

	private static List<Part> mediaToParts(Collection<Media> media) {
		List<Part> parts = new ArrayList<>();

		List<Part> mediaParts = media.stream()
			.map(mediaData -> PartMaker.fromMimeTypeAndData(mediaData.getMimeType().toString(), mediaData.getData()))
			.toList();

		if (!CollectionUtils.isEmpty(mediaParts)) {
			parts.addAll(mediaParts);
		}

		return parts;
	}

	private List<Tool> getFunctionTools(Set<String> functionNames) {
		final var tool = Tool.newBuilder();

		final List<FunctionDeclaration> functionDeclarations = this.resolveFunctionCallbacks(functionNames)
			.stream()
			.map(functionCallback -> FunctionDeclaration.newBuilder()
				.setName(functionCallback.getName())
				.setDescription(functionCallback.getDescription())
				.setParameters(jsonToSchema(functionCallback.getInputTypeSchema()))
				.build())
			.toList();
		tool.addAllFunctionDeclarations(functionDeclarations);
		return List.of(tool.build());
	}

	private static String structToJson(Struct struct) {
		try {
			return JsonFormat.printer().print(struct);
		}
		catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	private static Struct jsonToStruct(String json) {
		try {
			var structBuilder = Struct.newBuilder();
			JsonFormat.parser().ignoringUnknownFields().merge(json, structBuilder);
			return structBuilder.build();
		}
		catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	private static Schema jsonToSchema(String json) {
		try {
			var schemaBuilder = Schema.newBuilder();
			JsonFormat.parser().ignoringUnknownFields().merge(json, schemaBuilder);
			return schemaBuilder.build();
		}
		catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	private GenerateContentResponse getContentResponse(AnthropicRequest request) {
		try {
			return request.model.generateContent(request.contents);
		}
		catch (Exception e) {
			throw new RuntimeException("Failed to generate content", e);
		}
	}

	@Override
	public ChatOptions getDefaultOptions() {
		return VertexAIAnthropicChatOptions.fromOptions(this.defaultOptions);
	}

}
