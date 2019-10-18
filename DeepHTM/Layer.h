#pragma once

#include"GL.h"

#include<iostream>
#include<algorithm>

namespace DeepHTM
{
	namespace Layer
	{
		struct Config
		{
			GLuint minibatchSize;
			GLfloat learningRate;
		};

		class Layer
		{
		protected:
			const Config* config;

		public:
			Layer(const Config& config) : config(&config)
			{

			}

			virtual ~Layer()
			{

			}

			const Config& GetConfig() const
			{
				return *config;
			}
		};

		class Activation : public Layer
		{
		protected:
			GLuint outputCount;

			GL::ComputeShader evaluation, gradientEvaluation;

		public:
			Activation(const Config& config, GLuint outputCount, GL::ComputeShader&& evaluation, GL::ComputeShader&& gradientEvaluation) :
				Layer(config),
				outputCount(outputCount),
				evaluation(std::move(evaluation)),
				gradientEvaluation(std::move(gradientEvaluation))
			{

			}

			void Evaluate(GL::ShaderStorageBuffer<GLfloat>& outputs)
			{
				evaluation.Use();

				outputs.Bind(0);

				glDispatchCompute(outputCount, config->minibatchSize, 1);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			}

			void EvaluateGradients(const GL::ShaderStorageBuffer<GLfloat>& outputs, GL::ShaderStorageBuffer<GLfloat>& gradients)
			{
				gradientEvaluation.Use();

				outputs.Bind(0);
				gradients.Bind(1);

				glDispatchCompute(outputCount, config->minibatchSize, 1);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			}
		};

		//Leaky ReLU
		class ReLU : public Activation
		{
		public:
			ReLU(const Config& config, GLuint outputCount) :
				Activation(config, outputCount, GL::ComputeShader("shaders/relu_evaluation.comp"), GL::ComputeShader("shaders/relu_gradient_evaluation.comp"))
			{

			}
		};

		class Sigmoid : public Activation
		{
		public:
			Sigmoid(const Config& config, GLuint outputCount) :
				Activation(config, outputCount, GL::ComputeShader("shaders/sigmoid_evaluation.comp"), GL::ComputeShader("shaders/sigmoid_gradient_evaluation.comp"))
			{

			}
		};

		class MSE : public Layer
		{
		private:
			GLuint outputCount;
			
			GL::ComputeShader gradientEvaluation;

		public:
			MSE(const Config& config, GLuint outputCount) :
				Layer(config),
				outputCount(outputCount),
				gradientEvaluation("shaders/mse_gradient_evaluation.comp")
			{

			}

			void EvaluateGradients(const GL::ShaderStorageBuffer<GLfloat>& targets, GLintptr targetsOffset, const GL::ShaderStorageBuffer<GLfloat>& outputs, GL::ShaderStorageBuffer<GLfloat>& gradients)
			{
				gradientEvaluation.Use();

				targets.Bind(0, targetsOffset, (GLsizeiptr)outputCount * config->minibatchSize);
				outputs.Bind(1);
				gradients.Bind(2);

				glDispatchCompute(outputCount, config->minibatchSize, 1);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			}
		};

		class Transform : public Layer
		{
		protected:
			GLuint inputCount, outputCount;
			GLuint totalOutputCount;

			GL::ShaderStorageBuffer<GLfloat> outputs, gradients;
			GL::ShaderStorageBuffer<GLfloat> weights, biases;

			GL::ComputeShader evaluation, gradientEvaluation, update;

		public:
			Transform(const Config& config, GLuint inputCount, GLuint outputCount, GL::ComputeShader&& evaluation, GL::ComputeShader&& gradientEvaluation, GL::ComputeShader&& update) :
				Layer(config),
				inputCount(inputCount), outputCount(outputCount),
				totalOutputCount(outputCount* config.minibatchSize),
				outputs(totalOutputCount), gradients(totalOutputCount),
				weights((GLsizeiptr)inputCount* outputCount), biases(outputCount),
				evaluation(std::move(evaluation)),
				gradientEvaluation(std::move(gradientEvaluation)),
				update(std::move(update))
			{
				const float range = sqrtf(2.f / inputCount);
				weights.Randomize(range);
				biases.Randomize(range);

				//biases.SetData([]() { return 0.f; });

				gradients.SetData([]() { return 0.f; });
			}

			Transform(const Config& config, GLuint inputCount, GLuint outputCount) :
				Transform(config, inputCount, outputCount,
					{
						"shaders/transform_evaluation.comp",

						"#define EXTERNAL_PARAMETERS\n"
						"#define INPUTS_BINDING 0\n"
						"#define OUTPUTS_BINDING 1\n"
						"#define WEIGHTS_BINDING 2\n"
						"#define BIASES_BINDING 3\n"
						"#define INPUT_COUNT " + std::to_string(inputCount) + "\n"
					},
					{
						"shaders/transform_gradient_evaluation.comp",

						"#define EXTERNAL_PARAMETERS\n"
						"#define OUTPUT_COUNT " + std::to_string(outputCount) + "\n"
					},
					{
						"shaders/transform_update.comp",

						"#define EXTERNAL_PARAMETERS\n"
						"#define LEARNING_RATE_LOCATION 0\n"
						"#define INPUT_COUNT_LOCATION 1\n"
						"#define GRADIENTS_BINDING 0\n"
						"#define INPUTS_BINDING 1\n"
						"#define WEIGHTS_BINDING 2\n"
						"#define BIASES_BINDING 3\n"
						"#define MINIBATCH_SIZE " + std::to_string(config.minibatchSize) + "\n"
					})
			{
				
			}

			GLuint GetOutputCount() const
			{
				return outputCount;
			}

			GLuint GetTotalOutputCount() const
			{
				return totalOutputCount;
			}

			GL::ShaderStorageBuffer<GLfloat>& GetOutputs()
			{
				return outputs;
			}

			const GL::ShaderStorageBuffer<GLfloat>& GetOutputs() const
			{
				return outputs;
			}

			GL::ShaderStorageBuffer<GLfloat>& GetGradients()
			{
				return gradients;
			}

			const GL::ShaderStorageBuffer<GLfloat>& GetWeights() const
			{
				return weights;
			}

			void Evaluate(const GL::ShaderStorageBuffer<GLfloat>& inputs, GLintptr inputsOffset)
			{
				const GLsizeiptr inputsLength = (GLsizeiptr)inputCount * config->minibatchSize;

				evaluation.Use();

				inputs.Bind(0, inputsOffset, inputsLength);
				outputs.Bind(1);
				weights.Bind(2);
				biases.Bind(3);

				glDispatchCompute(outputCount, config->minibatchSize, 1);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			}

			void EvaluateGradients(GL::ShaderStorageBuffer<GLfloat>& inputGradients)
			{
				gradientEvaluation.Use();
					
				inputGradients.Bind(0);
				gradients.Bind(1);
				weights.Bind(2);

				glDispatchCompute(inputCount, config->minibatchSize, 1);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			}

			void Update(const GL::ShaderStorageBuffer<GLfloat>& inputs, GLintptr inputsOffset)
			{
				const GLsizeiptr inputsLength = (GLsizeiptr)inputCount * config->minibatchSize;

				update.Use();

				glUniform1f(0, config->learningRate);
				glUniform1ui(1, inputCount);

				gradients.Bind(0);
				inputs.Bind(1, inputsOffset, inputsLength);
				weights.Bind(2);
				biases.Bind(3);

				glDispatchCompute(inputCount + 1u, outputCount, 1);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			}
		};

		class SparseTransform : public Transform
		{
		protected:
			GLuint activeInputCount;

		public:
			SparseTransform(const Config& config, GLuint inputCount, GLuint activeInputCount, GLuint outputCount) :
				Transform(config, inputCount, outputCount,
					{
						"shaders/sparse_transform_evaluation.comp",

						"#define EXTERNAL_PARAMETERS\n"
						"#define INPUT_COUNT " + std::to_string(inputCount) + "\n"
						"#define INDEX_COUNT " + std::to_string(activeInputCount) + "\n"
					},
					{
						"shaders/sparse_transform_gradient_evaluation.comp",

						"#define EXTERNAL_PARAMETERS\n"
						"#define INPUT_COUNT " + std::to_string(inputCount) + "\n"
						"#define OUTPUT_COUNT " + std::to_string(outputCount) + "\n"
					},
					{
						"shaders/sparse_transform_update.comp",

						"#define EXTERNAL_PARAMETERS\n"
						"#define MINIBATCH_SIZE " + std::to_string(config.minibatchSize) + "\n"
					}),
				activeInputCount(activeInputCount)
			{

			}

			void Evaluate(const GL::ShaderStorageBuffer<GLfloat>& inputs, GLintptr inputsOffset, const GL::ShaderStorageBuffer<GLuint>& inputIndices)
			{
				const GLsizeiptr inputsLength = (GLsizeiptr)inputCount * config->minibatchSize;

				evaluation.Use();

				inputs.Bind(0, inputsOffset, inputsLength);
				inputIndices.Bind(1);
				outputs.Bind(2);
				weights.Bind(3);
				biases.Bind(4);

				glDispatchCompute(outputCount, config->minibatchSize, 1);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			}

			void EvaluateGradients(GL::ShaderStorageBuffer<GLfloat>& inputGradients, const GL::ShaderStorageBuffer<GLuint>& inputIndices)
			{
				gradientEvaluation.Use();

				inputGradients.Bind(0);
				inputIndices.Bind(1);
				gradients.Bind(2);
				weights.Bind(3);

				glDispatchCompute(activeInputCount, config->minibatchSize, 1);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			}

			void Update(const GL::ShaderStorageBuffer<GLfloat>& inputs, GLintptr inputsOffset, const GL::ShaderStorageBuffer<GLuint>& inputIndices)
			{
				const GLsizeiptr inputsLength = (GLsizeiptr)inputCount * config->minibatchSize;

				update.Use();

				glUniform1f(0, config->learningRate);
				glUniform1ui(1, inputCount);

				gradients.Bind(0);
				inputs.Bind(1, inputsOffset, inputsLength);
				inputIndices.Bind(2);
				weights.Bind(3);
				biases.Bind(4);

				glDispatchCompute(activeInputCount + 1u, outputCount, 1);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			}
		};

		class SpatialPooler : public Transform
		{
		private:
			GLuint winnerMinicolumnCount;
			GLfloat sparsity;
			GLfloat dutyCycleInertia, boostingWeight;

			GL::ShaderStorageBuffer<GLfloat> dutyCycles;
			GL::ShaderStorageBuffer<GLuint> winnerMinicolumns, masks;
			
			GL::ComputeShader kWinner, boosting;

			GLuint iteration;

		public:
			SpatialPooler(const Config& config, GLuint inputCount, GLuint minicolumnCount, GLuint winnerMinicolumnCount, GLfloat dutyCycleInertia, GLfloat boostingWeight) : 
				Transform(config, inputCount, minicolumnCount),
				winnerMinicolumnCount(winnerMinicolumnCount),
				sparsity(static_cast<GLfloat>(winnerMinicolumnCount) / minicolumnCount),
				dutyCycleInertia(dutyCycleInertia), boostingWeight(boostingWeight),
				dutyCycles(totalOutputCount),
				winnerMinicolumns((GLsizeiptr)winnerMinicolumnCount * config.minibatchSize), masks(totalOutputCount),
				kWinner
				(
					"shaders/sp_k_winner.comp",

					"#define EXTERNAL_PARAMETERS\n"
					"#define SPARSITY_LOCATION 0\n"
					"#define MINICOLUMNS_BINDING 0\n"
					"#define DUTY_CYCLES_BINDING 1\n"
					"#define MASKS_BINDING 2\n"
					"#define WINNER_MINICOLUMNS_BINDING 3\n"
					"#define MINICOLUMN_COUNT " + std::to_string(minicolumnCount) + "\n"
					"#define WINNER_MINICOLUMN_COUNT " + std::to_string(winnerMinicolumnCount)
				),
				boosting
				(
					"shaders/sp_boosting.comp",

					"#define EXTERNAL_PARAMETERS\n"
					"#define SPARSITY_LOCATION 0\n"
					"#define DUTY_CYCLE_INERTIA_LOCATION 1\n"
					"#define BOOSTING_WEIGHT_LOCATION 2\n"
					"#define ITERATION_LOCATION 3\n"
					"#define DUTY_CYCLES_BINDING 0\n"
					"#define MINICOLUMNS_BINDING 1\n"
					"#define MASKS_BINDING 2\n"
					"#define GRADIENTS_BINDING 3\n"
					"#define MINIBATCH_SIZE " + std::to_string(config.minibatchSize) + "\n"
				),
				iteration(0u)
			{
				dutyCycles.SetData([=]() { return sparsity; });
			}

			SpatialPooler(const Config& config, GLuint inputCount, GLuint minicolumnCount, GLuint winnerMinicolumnCount) : SpatialPooler(config, inputCount, minicolumnCount, winnerMinicolumnCount, 0.99f, 1.f)
			{

			}

			GLuint GetMinicolumnCount() const
			{
				return outputCount;
			}

			GLuint GetTotalMinicolumnCount() const
			{
				return totalOutputCount;
			}

			GLuint GetWinnerMinicolumnCount() const
			{
				return winnerMinicolumnCount;
			}

			GLfloat GetSparsity() const
			{
				return sparsity;
			}

			GLfloat GetDutyCycleInertia() const
			{
				return dutyCycleInertia;
			}

			GLfloat GetBoostingWeight() const
			{
				return boostingWeight;
			}

			const GL::ShaderStorageBuffer<GLfloat>& GetMinicolumns() const
			{
				return outputs;
			}

			const GL::ShaderStorageBuffer<GLuint>& GetWinnerMinicolumns() const
			{
				return winnerMinicolumns;
			}

			const GL::ShaderStorageBuffer<GLfloat>& GetDutyCycles() const
			{
				return dutyCycles;
			}

			void SetDutyCycleInertia(GLfloat newDutyCycleInertia)
			{
				dutyCycleInertia = newDutyCycleInertia;
			}

			void SetBoostingWeight(GLfloat newBoostingWeight)
			{
				boostingWeight = newBoostingWeight;
			}

			void Evaluate(const GL::ShaderStorageBuffer<float>& inputs, GLintptr inputsOffset)
			{
				Transform::Evaluate(inputs, inputsOffset);

				const GLsizeiptr inputsLength = (GLsizeiptr)inputCount * config->minibatchSize;

				kWinner.Use();

				glUniform1f(0, sparsity);

				outputs.Bind(0);
				dutyCycles.Bind(1);
				masks.Bind(2);
				winnerMinicolumns.Bind(3);

				glDispatchCompute(config->minibatchSize, 1, 1);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			}

			void EvaluateGradients()
			{
				boosting.Use();

				glUniform1f(0, sparsity);
				glUniform1f(1, dutyCycleInertia);
				glUniform1f(2, boostingWeight);
				glUniform1ui(3, ++iteration);

				dutyCycles.Bind(0);
				outputs.Bind(1);
				masks.Bind(2);
				gradients.Bind(3);

				glDispatchCompute(outputCount, config->minibatchSize, 1);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			}

			void EvaluateGradients(GL::ShaderStorageBuffer<GLfloat>& inputGradients)
			{
				EvaluateGradients();

				Transform::EvaluateGradients(inputGradients);
			}
		};
	}
}