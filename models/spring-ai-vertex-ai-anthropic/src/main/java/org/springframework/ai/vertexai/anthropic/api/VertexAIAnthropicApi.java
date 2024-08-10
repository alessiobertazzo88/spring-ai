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
package org.springframework.ai.vertexai.anthropic.api;

import com.google.auth.oauth2.GoogleCredentials;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.ai.vertexai.anthropic.model.*;
import org.springframework.ai.vertexai.anthropic.model.stream.EventType;
import org.springframework.ai.vertexai.anthropic.model.stream.StreamEvent;
import org.springframework.ai.vertexai.anthropic.model.stream.ToolUseAggregationEvent;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.Assert;
import org.springframework.web.client.RestClient;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Vertex AI Anthropic API client.
 *
 * @author Alessio Bertazzo
 * @since 1.0.0
 */
public class VertexAIAnthropicApi {

	private final String projectId;

	private final String location;

	private final GoogleCredentials credentials;

	private final RestClient restClient;

	private final WebClient webClient;

	private final StreamHelper streamHelper = new StreamHelper();

	private static final String VERTEXAI_BASE_URL = "https://%s-aiplatform.googleapis.com";

	private static final String VERTEXAI_ANTHROPIC_ENDPOINT = "/v1/projects/%s/locations/%s/publishers/anthropic/models/%s";

	private static final Predicate<String> SSE_DONE_PREDICATE = "[DONE]"::equals;

	private VertexAIAnthropicApi(String projectId, String location, GoogleCredentials credentials) {
		this.projectId = projectId;
		this.location = location;
		this.credentials = credentials;

		Consumer<HttpHeaders> jsonContentHeaders = headers -> {
			headers.setContentType(MediaType.APPLICATION_JSON);
		};

		this.restClient = RestClient.builder()
			.baseUrl(VERTEXAI_BASE_URL.formatted(location))
			.defaultHeaders(jsonContentHeaders)
			.defaultStatusHandler(RetryUtils.DEFAULT_RESPONSE_ERROR_HANDLER)
			.build();

		this.webClient = WebClient.builder()
			.baseUrl(VERTEXAI_BASE_URL.formatted(location))
			.defaultHeaders(jsonContentHeaders)
			.defaultStatusHandler(HttpStatusCode::isError,
					resp -> Mono.just(new RuntimeException("Response exception, Status: [" + resp.statusCode()
							+ "], Body:[" + resp.bodyToMono(java.lang.String.class) + "]")))
			.build();
	}

	// create the builder class
	public static class Builder {

		private String projectId;

		private String location;

		private GoogleCredentials credentials;

		public Builder projectId(String projectId) {
			this.projectId = projectId;
			return this;
		}

		public Builder location(String location) {
			this.location = location;
			return this;
		}

		public Builder credentials(GoogleCredentials credentials) {
			this.credentials = credentials;
			return this;
		}

		public VertexAIAnthropicApi build() {
			return new VertexAIAnthropicApi(projectId, location, credentials);
		}

	}

	/**
	 * Creates a model response for the given chat conversation.
	 * @param chatRequest The chat completion request.
	 * @return Entity response with {@link ChatCompletionResponse} as a body and HTTP
	 * status code and headers.
	 */
	public ResponseEntity<ChatCompletionResponse> chatCompletion(ChatCompletionRequest chatRequest, String model) {

		Assert.notNull(chatRequest, "The request body can not be null.");
		Assert.isTrue(!chatRequest.stream(), "Request must set the steam property to false.");

		return this.restClient.post()
			.uri(VERTEXAI_ANTHROPIC_ENDPOINT.formatted(projectId, location, model) + ":rawPredict")
			.headers(headers -> headers.setBearerAuth(this.credentials.getAccessToken().getTokenValue()))
			.body(chatRequest)
			.retrieve()
			.toEntity(ChatCompletionResponse.class);
	}

	/**
	 * Creates a streaming chat response for the given chat conversation.
	 * @param chatRequest The chat completion request. Must have the stream property set
	 * to true.
	 * @return Returns a {@link Flux} stream from chat completion chunks.
	 */
	public Flux<ChatCompletionResponse> chatCompletionStream(ChatCompletionRequest chatRequest, String model) {

		Assert.notNull(chatRequest, "The request body can not be null.");
		Assert.isTrue(chatRequest.stream(), "Request must set the steam property to true.");

		AtomicBoolean isInsideTool = new AtomicBoolean(false);

		AtomicReference<StreamHelper.ChatCompletionResponseBuilder> chatCompletionReference = new AtomicReference<>();

		return this.webClient.post()
			.uri(VERTEXAI_ANTHROPIC_ENDPOINT.formatted(projectId, location, model) + ":streamRawPredict")
			.body(Mono.just(chatRequest), ChatCompletionRequest.class)
			.retrieve()
			.bodyToFlux(String.class)
			.takeUntil(SSE_DONE_PREDICATE)
			.filter(SSE_DONE_PREDICATE.negate())
			.map(content -> ModelOptionsUtils.jsonToObject(content, StreamEvent.class))
			.filter(event -> event.type() != EventType.PING)
			// Detect if the chunk is part of a streaming function call.
			.map(event -> {
				if (this.streamHelper.isToolUseStart(event)) {
					isInsideTool.set(true);
				}
				return event;
			})
			// Group all chunks belonging to the same function call.
			.windowUntil(event -> {
				if (isInsideTool.get() && this.streamHelper.isToolUseFinish(event)) {
					isInsideTool.set(false);
					return true;
				}
				return !isInsideTool.get();
			})
			// Merging the window chunks into a single chunk.
			.concatMapIterable(window -> {
				Mono<StreamEvent> monoChunk = window.reduce(new ToolUseAggregationEvent(),
						this.streamHelper::mergeToolUseEvents);
				return List.of(monoChunk);
			})
			.flatMap(mono -> mono)
			.map(event -> streamHelper.eventToChatCompletionResponse(event, chatCompletionReference))
			.filter(chatCompletionResponse -> chatCompletionResponse.type() != null);
	}

	// create a method that given the credentials variable return the bearer token. Check
	// that the token exists and is not expired before returning it and if it is expired,
	// refresh it.
	public String getBearerToken(GoogleCredentials credentials) {
		Assert.notNull(credentials, "The credentials can not be null.");

		try {
			// if (credentials.getAccessToken() == null) {
			// credentials.refresh();
			// } else {
			credentials.refreshIfExpired();
			// }
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}

		return credentials.getAccessToken().getTokenValue();
	}

}
