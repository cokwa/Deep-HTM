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
		public:
			Layer()
			{

			}

			virtual ~Layer()
			{

			}
		};

		class SpatialPooler : public Layer
		{
		private:
			enum Location
			{
				InputCount,
				MinicolumnCount,
				Sparsity,
				DutyCycleInertia,
				BoostingWeight,
				LearningRate,
				Iteration
			};

			enum Binding
			{
				Inputs,
				Minicolumns,
				MinicolumnStates,
				DutyCycles,
				Gradients,
				Weights,
				Biases,
				WinnerMinicolumns,
			};

			const Config& config;

			const GLuint inputCount, minicolumnsSizeX, minicolumnsSizeY, minicolumnCount, winnerMinicolumnCount, totalMinicolumnCount;
			const GLfloat sparsity;
			GLfloat dutyCycleInertia, boostingWeight;

			GL::ShaderStorageBuffer<GLfloat> minicolumns, dutyCycles, gradients, weights, biases;
			GL::ShaderStorageBuffer<GLuint> minicolumnStates/*TODO: probably needs a better name like 'masks' or something*/, winnerMinicolumns;

			GL::ComputeShader fullyConnected, kWinner, boosting, weightUpdate;

			GLuint iteration;

		public:
			SpatialPooler(const Config& config, GLuint inputCount, GLuint minicolumnsSizeX, GLuint minicolumnsSizeY, GLuint winnerMinicolumnCount, GLfloat dutyCycleInertia, GLfloat boostingWeight) : config(config), inputCount(inputCount), minicolumnsSizeX(minicolumnsSizeX), minicolumnsSizeY(minicolumnsSizeY), minicolumnCount(minicolumnsSizeX * minicolumnsSizeY), winnerMinicolumnCount(winnerMinicolumnCount), totalMinicolumnCount(minicolumnCount * config.minibatchSize), sparsity(static_cast<GLfloat>(winnerMinicolumnCount) / minicolumnCount), dutyCycleInertia(dutyCycleInertia), boostingWeight(boostingWeight), minicolumns(totalMinicolumnCount), dutyCycles(totalMinicolumnCount), gradients(totalMinicolumnCount), weights(inputCount * minicolumnCount), biases(minicolumnCount), minicolumnStates(totalMinicolumnCount), winnerMinicolumns(winnerMinicolumnCount * config.minibatchSize),
				fullyConnected
				(
					"shaders/sp_fully_connected.comp",

					"#define EXTERNAL_PARAMETERS\n"
					"#define INPUT_COUNT_LOCATION " + std::to_string(InputCount) + "\n"
					"#define INPUTS_BINDING " + std::to_string(Inputs) + "\n"
					"#define MINICOLUMNS_BINDING " + std::to_string(Minicolumns) + "\n"
					"#define WEIGHTS_BINDING " + std::to_string(Weights) + "\n"
					"#define BIASES_BINDING " + std::to_string(Biases) + "\n"
				),
				kWinner
				(
					"shaders/sp_k_winner.comp",

					"#define EXTERNAL_PARAMETERS\n"
					"#define SPARSITY_LOCATION " + std::to_string(Sparsity) + "\n"
					"#define MINICOLUMNS_BINDING " + std::to_string(Minicolumns) + "\n"
					"#define DUTY_CYCLES_BINDING " + std::to_string(DutyCycles) + "\n"
					"#define MINICOLUMN_STATES_BINDING " + std::to_string(MinicolumnStates) + "\n"
					"#define WINNER_MINICOLUMNS_BINDING " + std::to_string(WinnerMinicolumns) + "\n"
					"#define MINICOLUMNS_SIZE_X " + std::to_string(minicolumnsSizeX) + "\n"
					"#define MINICOLUMNS_SIZE_Y " + std::to_string(minicolumnsSizeY) + "\n"
					"#define MINICOLUMN_COUNT " + std::to_string(minicolumnCount) + "\n"
					"#define WINNER_MINICOLUMN_COUNT " + std::to_string(winnerMinicolumnCount)
				),
				boosting
				(
					"shaders/sp_boosting.comp",

					"#define EXTERNAL_PARAMETERS\n"
					"#define SPARSITY_LOCATION " + std::to_string(Sparsity) + "\n"
					"#define DUTY_CYCLE_INERTIA_LOCATION " + std::to_string(DutyCycleInertia) + "\n"
					"#define BOOSTING_WEIGHT_LOCATION " + std::to_string(BoostingWeight) + "\n"
					"#define ITERATION_LOCATION " + std::to_string(Iteration) + "\n"
					"#define DUTY_CYCLES_BINDING " + std::to_string(DutyCycles) + "\n"
					"#define MINICOLUMN_STATES_BINDING " + std::to_string(MinicolumnStates) + "\n"
					"#define GRADIENTS_BINDING " + std::to_string(Gradients) + "\n"
					"#define MINIBATCH_SIZE " + std::to_string(config.minibatchSize) + "\n"
				),
				weightUpdate
				{
					"shaders/sp_weight_update.comp",

					"#define EXTERNAL_PARAMETERS\n"
					"#define LEARNING_RATE_LOCATION " + std::to_string(LearningRate) + "\n"
					"#define INPUT_COUNT_LOCATION " + std::to_string(InputCount) + "\n"
					"#define GRADIENTS_BINDING " + std::to_string(Gradients) + "\n"
					"#define INPUTS_BINDING " + std::to_string(Inputs) + "\n"
					"#define WEIGHTS_BINDING " + std::to_string(Weights) + "\n"
					"#define BIASES_BINDING " + std::to_string(Biases) + "\n"
					"#define MINIBATCH_SIZE " + std::to_string(config.minibatchSize) + "\n"
				},
				iteration(0u)
			{
				dutyCycles.SetData([=]() { return sparsity; });

				weights.Randomize();
				biases.Randomize();

				//biases.SetData([]() { return 0.f; });

				gradients.SetData([]() { return 0.f; });
			}

			SpatialPooler(const Config& config, GLuint inputCount, GLuint minicolumnsSizeX, GLuint minicolumnsSizeY, GLuint winnerMinicolumnCount) : SpatialPooler(config, inputCount, minicolumnsSizeX, minicolumnsSizeY, winnerMinicolumnCount, 0.99f, 1.f)
			{

			}

			const Config& GetConfig() const
			{
				return config;
			}

			GLuint GetInputCount() const
			{
				return inputCount;
			}

			GLuint GetMinicolumnsSizeX() const
			{
				return minicolumnsSizeX;
			}

			GLuint GetMinicolumnsSizeY() const
			{
				return minicolumnsSizeY;
			}

			GLuint GetMinicolumnCount() const
			{
				return minicolumnCount;
			}

			GLuint GetTotalMinicolumnCount() const
			{
				return totalMinicolumnCount;
			}

			GLfloat GetDutyCycleInertia() const
			{
				return dutyCycleInertia;
			}

			GLfloat GetBoostingWeight() const
			{
				return boostingWeight;
			}

			const GL::ShaderStorageBuffer<GLfloat>& GetDutyCycles() const
			{
				return dutyCycles;
			}

			const GL::ShaderStorageBuffer<GLfloat>& GetWeights() const
			{
				return weights;
			}

			void SetDutyCycleInertia(GLfloat newDutyCycleInertia)
			{
				dutyCycleInertia = newDutyCycleInertia;
			}

			void SetBoostingWeight(GLfloat newBoostingWeight)
			{
				boostingWeight = newBoostingWeight;
			}

			void Run(const GL::ShaderStorageBuffer<float>& inputs, GLintptr inputsOffset, GLsizeiptr inputsLength)
			{
				fullyConnected.Use();
				{
					glUniform1ui(InputCount, inputCount);
					
					inputs.Bind(Inputs, inputsOffset, inputsLength);
					minicolumns.Bind(Minicolumns);
					weights.Bind(Weights);
					biases.Bind(Biases);

					glDispatchCompute(minicolumnCount, config.minibatchSize, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
				}

				kWinner.Use();
				{
					glUniform1f(Sparsity, sparsity);

					minicolumns.Bind(Minicolumns);
					dutyCycles.Bind(DutyCycles);
					minicolumnStates.Bind(MinicolumnStates);
					winnerMinicolumns.Bind(WinnerMinicolumns);

					glDispatchCompute(config.minibatchSize, 1, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
				}

				boosting.Use();
				{
					glUniform1f(Sparsity, sparsity);
					glUniform1f(DutyCycleInertia, dutyCycleInertia);
					glUniform1f(BoostingWeight, boostingWeight);
					glUniform1ui(Iteration, ++iteration);

					dutyCycles.Bind(DutyCycles);
					minicolumnStates.Bind(MinicolumnStates);
					gradients.Bind(Gradients);

					glDispatchCompute(minicolumnCount, config.minibatchSize, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
				}

				weightUpdate.Use();
				{
					glUniform1f(LearningRate, config.learningRate);
					glUniform1ui(InputCount, inputCount);

					gradients.Bind(Gradients);
					inputs.Bind(Inputs, inputsOffset, inputsLength);
					weights.Bind(Weights);
					biases.Bind(Biases);

					glDispatchCompute(inputCount + 1u, minicolumnCount, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
				}

				/*auto minicolumnsData = minicolumns.GetData();
				auto masksData = minicolumnStates.GetData();

				for (size_t i = 0; i < minicolumnsData.size(); i++)
				{
					std::cout << (masksData[i] ? minicolumnsData[i] : 0.f) << ' ';
				}

				std::cout << std::endl;*/
			}

			void Run(const GL::ShaderStorageBuffer<GLfloat>& inputs)
			{
				Run(inputs, 0, inputs.GetLength());
			}
		};
	}
}