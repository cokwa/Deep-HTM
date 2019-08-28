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
				MinicolumnCount
			};

			enum Binding
			{
				Inputs,
				Minicolumns,
				Weights,
				Biases,
				WinnerMinicolumns
			};

			const Config& config;

			const GLuint inputCount, minicolumnsSizeX, minicolumnsSizeY, minicolumnCount, winnerMinicolumnCount;

			GL::ShaderStorageBuffer<GLfloat> minicolumns, weights, biases;
			GL::ShaderStorageBuffer<GLuint> winnerMinicolumns;

			GL::ComputeShader fullyConnected, kWinner;

		public:
			SpatialPooler(const Config& config, GLuint inputCount, GLuint minicolumnsSizeX, GLuint minicolumnsSizeY, GLuint winnerMinicolumnCount) : config(config), inputCount(inputCount), minicolumnsSizeX(minicolumnsSizeX), minicolumnsSizeY(minicolumnsSizeY), minicolumnCount(minicolumnsSizeX * minicolumnsSizeY), winnerMinicolumnCount(winnerMinicolumnCount), minicolumns((GLsizeiptr)minicolumnCount * config.minibatchSize), weights((GLsizeiptr)inputCount * minicolumnCount), biases(minicolumnCount), winnerMinicolumns((GLsizeiptr)winnerMinicolumnCount * config.minibatchSize),
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
					"#define WINNER_MINICOLUMNS_BINDING " + std::to_string(WinnerMinicolumns) + "\n"
					"#define MINICOLUMNS_SIZE_X " + std::to_string(minicolumnsSizeX) + "\n"
					"#define MINICOLUMNS_SIZE_Y " + std::to_string(minicolumnsSizeY) + "\n"
					"#define MINICOLUMN_COUNT " + std::to_string(minicolumnCount) + "\n"
					"#define WINNER_MINICOLUMN_COUNT " + std::to_string(winnerMinicolumnCount)
				)
			{
				weights.Randomize();
				biases.Randomize();
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
					winnerMinicolumns.Bind(WinnerMinicolumns);

					glDispatchCompute(config.minibatchSize, 1, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
				}
			}
		};
	}
}