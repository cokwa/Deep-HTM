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

			GL::ShaderStorageBuffer<GLfloat> inputs(100);
			inputs.Randomize();

			Layer::SpatialPooler* sp = new Layer::SpatialPooler(config, inputs.GetSize(), 32, 32, 40);
			sp->Run(inputs);
			
			delete sp;
		}

		virtual ~DeepHTM()
		{
		}
	};
}