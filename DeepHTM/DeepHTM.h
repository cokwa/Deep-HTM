#pragma once

#include"GL.h"
#include"Layer.h"

namespace DeepHTM
{
	class DeepHTM
	{
	public:
		DeepHTM()
		{
			GL::ComputeShader* computeShader = new GL::ComputeShader("shaders/sp_fully_connected.comp");
			delete computeShader;

			computeShader = new GL::ComputeShader("shaders/sp_k_winner.comp");
			delete computeShader;
		}

		virtual ~DeepHTM()
		{
		}
	};
}