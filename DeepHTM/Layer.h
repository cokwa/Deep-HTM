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
				BoostingWeight
			};

			enum Binding
			{
				Inputs,
				Minicolumns,
				MinicolumnStates,
				DutyCycles,
				Deltas,
				Weights,
				Biases,
				WinnerMinicolumns,
			};

			const Config& config;

			const GLuint inputCount, minicolumnsSizeX, minicolumnsSizeY, minicolumnCount, winnerMinicolumnCount, totalMinicolumnCount;
			const GLfloat sparsity;
			GLfloat dutyCycleInertia, boostingWeight;

			GL::ShaderStorageBuffer<GLfloat> minicolumns, dutyCycles, deltas, weights, biases;
			GL::ShaderStorageBuffer<GLuint> minicolumnStates, winnerMinicolumns;

			GL::ComputeShader fullyConnected, kWinner, boosting;

		public:
			SpatialPooler(const Config& config, GLuint inputCount, GLuint minicolumnsSizeX, GLuint minicolumnsSizeY, GLuint winnerMinicolumnCount, GLfloat dutyCycleInertia, GLfloat boostingWeight) : config(config), inputCount(inputCount), minicolumnsSizeX(minicolumnsSizeX), minicolumnsSizeY(minicolumnsSizeY), minicolumnCount(minicolumnsSizeX * minicolumnsSizeY), winnerMinicolumnCount(winnerMinicolumnCount), totalMinicolumnCount((GLsizeiptr)minicolumnCount * config.minibatchSize), sparsity(static_cast<GLfloat>(winnerMinicolumnCount) / minicolumnCount), dutyCycleInertia(dutyCycleInertia), boostingWeight(boostingWeight), minicolumns(totalMinicolumnCount), dutyCycles(minicolumnCount), deltas(minicolumnCount), weights((GLsizeiptr)inputCount* minicolumnCount), biases(minicolumnCount), minicolumnStates(totalMinicolumnCount), winnerMinicolumns((GLsizeiptr)winnerMinicolumnCount* config.minibatchSize),
				fullyConnected
				(
					"shaders/sp_fully_connected.comp",

					"#define EXTERNAL_PARAMETERS\n"
					"#define INPUT_COUNT_LOCATION " + std::to_string(InputCount) + "\n"
					"#define MINICOLUMN_COUNT_LOCATION " + std::to_string(MinicolumnCount) + "\n"
					"#define INPUTS_BINDING " + std::to_string(Inputs) + "\n"
					"#define MINICOLUMNS_BINDING " + std::to_string(Minicolumns) + "\n"
					"#define WEIGHTS_BINDING " + std::to_string(Weights) + "\n"
					"#define BIASES_BINDING " + std::to_string(Biases) + "\n"
				),
				kWinner
				(
					"shaders/sp_k_winner.comp",

					"#define EXTERNAL_PARAMETERS\n"
					"#define MINICOLUMNS_BINDING " + std::to_string(Minicolumns) + "\n"
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
					"#define BOOSTING_WEIGHT " + std::to_string(BoostingWeight) + "\n"
					"#define DUTY_CYCLES_BINDING " + std::to_string(DutyCycles) + "\n"
					"#define MINICOLUMN_STATES_BINDING " + std::to_string(MinicolumnStates) + "\n"
					"#define DELTAS_BINDING " + std::to_string(Deltas) + "\n"
					"#define MINIBATCH_SIZE " + std::to_string(config.minibatchSize) + "\n"
				)
			{
				dutyCycles.SetData([=]() { return sparsity; });

				weights.Randomize();
				biases.Randomize();
			}

			SpatialPooler(const Config& config, GLuint inputCount, GLuint minicolumnsSizeX, GLuint minicolumnsSizeY, GLuint winnerMinicolumnCount) : SpatialPooler(config, inputCount, minicolumnsSizeX, minicolumnsSizeY, winnerMinicolumnCount, 0.9f, 1.f)
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

			GLfloat GetDutyCycleInertia() const
			{
				return dutyCycleInertia;
			}

			GLfloat GetBoostingWeight() const
			{
				return boostingWeight;
			}

			void SetDutyCycleInertia(GLfloat newDutyCycleInertia)
			{
				dutyCycleInertia = newDutyCycleInertia;
			}

			void SetBoostingWeight(GLfloat newBoostingWeight)
			{
				boostingWeight = newBoostingWeight;
			}

			void Run(const GL::ShaderStorageBuffer<float>& inputs)
			{
				fullyConnected.Use();
				{
					glUniform1ui(InputCount, inputCount);
					glUniform1ui(MinicolumnCount, minicolumnCount);

					inputs.Bind(Inputs);
					minicolumns.Bind(Minicolumns);
					weights.Bind(Weights);
					biases.Bind(Biases);

					glDispatchCompute(minicolumnCount, config.minibatchSize, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
				}

				kWinner.Use();
				{
					minicolumns.Bind(Minicolumns);
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

					dutyCycles.Bind(DutyCycles);
					minicolumnStates.Bind(MinicolumnStates);
					deltas.Bind(Deltas);

					glDispatchCompute(minicolumnCount, 1, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
				}
			}
		};
	}
}