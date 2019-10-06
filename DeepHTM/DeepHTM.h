#pragma once

#include"GL.h"
#include"Layer.h"

namespace DeepHTM
{
	class DeepHTM
	{
	private:
		Layer::SpatialPooler spatialPooler;

	public:
		DeepHTM(const Layer::Config& config) : spatialPooler(config, 28 * 28, 1024, 20)
		{
			/*std::ifstream images("samples/train-images.idx3-ubyte", std::ifstream::binary);
			if (!images)
			{
				throw std::runtime_error("images");
			}

			std::ifstream labels("samples/train-labels.idx1-ubyte", std::ifstream::binary);
			if (!labels)
			{
				throw std::runtime_error("labels");
			}

			images.seekg(16, images.beg);
			labels.seekg(8, labels.beg);

			const GLuint inputWidth = 28, inputHeight = 28, inputCount = 60000;
			const GLsizeiptr inputMinibatchSize = inputWidth * inputHeight * config.minibatchSize;

			std::vector<GLubyte> buffer(inputWidth * inputHeight * inputCount);
			images.read(reinterpret_cast<char*>(&buffer[0]), buffer.size());

			GL::ShaderStorageBuffer<GLfloat> inputs(buffer.size());
			auto nextPixel = buffer.begin();
			inputs.SetData([&]() { return *(nextPixel++) / 255.f; });

			images.close();
			labels.close();

			for (GLuint minibatch = 0; minibatch < 100; minibatch++)
			{
				spatialPooler.Run(inputs, minibatch * inputMinibatchSize, (minibatch + 1) * inputMinibatchSize);
			}*/
		}

		virtual ~DeepHTM()
		{
		}

		const Layer::SpatialPooler& GetSpatialPooler() const
		{
			return spatialPooler;
		}
	};
}