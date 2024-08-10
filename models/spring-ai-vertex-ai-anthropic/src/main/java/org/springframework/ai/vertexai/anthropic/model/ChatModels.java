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
package org.springframework.ai.vertexai.anthropic.model;

import org.springframework.ai.model.ChatModelDescription;

/**
 * Check the <a href=
 * "https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#model-list">Use
 * Claude models </a> and <a href=
 * "https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#available-claude-models">Available
 * Anthropic Claude models </a> for additional details and options.
 *
 * @author Alessio Bertazzo
 * @since 1.0.0
 */
public enum ChatModels implements ChatModelDescription {

	// @formatter:off
	CLAUDE_3_5_SONNET("claude-3-5-sonnet@20240620"),

	CLAUDE_3_OPUS("claude-3-opus@20240229"),
	CLAUDE_3_SONNET("claude-3-sonnet@20240229"),
	CLAUDE_3_HAIKU("claude-3-haiku@20240307");
	// @formatter:on

	public final String value;

	ChatModels(String value) {
		this.value = value;
	}

	public String getValue() {
		return this.value;
	}

	@Override
	public String getName() {
		return this.value;
	}

}