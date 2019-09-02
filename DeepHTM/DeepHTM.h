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
			Layer::Config config;
			config.minibatchSize = 32;
			config.learningRate = 1e-2f;

			Layer::SpatialPooler* sp = new Layer::SpatialPooler(config, 100, 32, 32, 40);

			GL::ShaderStorageBuffer<GLfloat> inputs((GLsizeiptr)sp->GetInputCount() * config.minibatchSize);
			inputs.Randomize();

			sp->Run(inputs);
			
			delete sp;
		}

		virtual ~DeepHTM()
		{
		}
	};
}