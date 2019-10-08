#include"DeepHTM.h"

#include <SFML/Graphics.hpp>

#if _DEBUG

#pragma comment(lib, "sfml-system-d.lib")
#pragma comment(lib, "sfml-window-d.lib")
#pragma comment(lib, "sfml-graphics-d.lib")
#pragma comment(lib, "sfml-audio-d.lib")
#pragma comment(lib, "sfml-network-d.lib")

#else

#pragma comment(lib, "sfml-system.lib")
#pragma comment(lib, "sfml-window.lib")
#pragma comment(lib, "sfml-graphics.lib")
#pragma comment(lib, "sfml-audio.lib")
#pragma comment(lib, "sfml-network.lib")

#endif

#include<iostream>

int main()
{
	sf::RenderWindow window(sf::VideoMode(800, 800), "DeepHTM");
	
	window.setFramerateLimit(0);

	if (!gladLoadGL())
	{
		throw std::exception();
	}

	sf::Shader visualizer;
	visualizer.loadFromFile("shaders/debug.vert", "shaders/debug.frag");

	GLfloat vertices[]
	{
		-1.f, -1.f, 0.f, 1.f,
		0.f, 0.f,

		1.f, -1.f, 0.f, 1.f,
		1.f, 0.f,

		-1.f, 1.f, 0.f, 1.f,
		0.f, 1.f,

		1.f, 1.f, 0.f, 1.f,
		1.f, 1.f
	};

	GLuint vao, vbo;
	vao = vbo = 0;

	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);

	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, (const GLvoid*)0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, (const GLvoid*)(sizeof(GLfloat) * 4));

	glBindVertexArray(0);

	DeepHTM::Layer::SpatialPooler* spatialPooler = nullptr;
	DeepHTM::Layer::Linear* linear = nullptr;
	DeepHTM::Layer::Sigmoid* sigmoid = nullptr;
	DeepHTM::Layer::MSE* mse = nullptr;

	DeepHTM::Layer::Config config;
	config.minibatchSize = 32;
	config.learningRate = 1e-1f;

	const GLuint inputWidth = 28, inputHeight = 28, inputCount = 60000;
	const GLsizeiptr inputMinibatchSize = (GLsizeiptr)inputWidth * inputHeight * config.minibatchSize;

	try
	{
		spatialPooler = new DeepHTM::Layer::SpatialPooler(config, inputWidth * inputHeight, 1024, 20);
		linear = new DeepHTM::Layer::Linear(config, spatialPooler->GetMinicolumnCount(), inputWidth * inputHeight);
		sigmoid = new DeepHTM::Layer::Sigmoid(config, linear->GetOutputCount());
		mse = new DeepHTM::Layer::MSE(config, linear->GetOutputCount());
	}
	catch (const std::exception& exception)
	{
		std::cout << exception.what() << std::endl;
	}

	std::ifstream images("samples/train-images.idx3-ubyte", std::ifstream::binary);
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

	std::vector<GLubyte> buffer(inputWidth * inputHeight * inputCount);
	images.read(reinterpret_cast<char*>(&buffer[0]), buffer.size());

	DeepHTM::GL::ShaderStorageBuffer<GLfloat> inputs(buffer.size());
	auto nextPixel = buffer.begin();
	
	inputs.SetData([&]() { return *(nextPixel++) / 255.f; });

	images.close();
	labels.close();

	//{
	//	using namespace DeepHTM::Layer;

	//	Linear* layer1 = new Linear(config, inputWidth * inputHeight, 32);
	//	Activation* layer2 = new ReLU(config, 32);
	//	Linear* layer3 = new Linear(config, 32, inputWidth * inputHeight);
	//	MSE* layer4 = new MSE(config, inputWidth * inputHeight);

	//	std::vector<GLfloat> data(layer3->GetGradients().GetLength());
	//	GLfloat totalError = 0.f;

	//	for (int iteration = 0; /*iteration < 10000*/; iteration++)
	//	{
	//		GLsizeiptr inputsOffset = (GLsizeiptr)(iteration % (inputCount / config.minibatchSize))* inputMinibatchSize;
	//		
	//		layer1->Evaluate(inputs, inputsOffset);
	//		layer2->Evaluate(layer1->GetOutputs());
	//		layer3->Evaluate(layer1->GetOutputs(), 0);
	//		
	//		layer4->EvaluateGradients(inputs, inputsOffset, layer3->GetOutputs(), layer3->GetGradients());

	//		layer3->GetGradients().GetData(data.begin());

	//		for (GLfloat error : data)
	//		{
	//			totalError += pow(error, 2.f);
	//		}

	//		if (iteration % 100 == 99)
	//		{
	//			std::cout << 0.5f * totalError << std::endl;
	//			totalError = 0.f;
	//		}

	//		layer3->EvaluateGradients(layer1->GetGradients());
	//		layer2->EvaluateGradients(layer1->GetOutputs(), layer1->GetGradients());

	//		layer1->Update(inputs, inputsOffset);
	//		layer3->Update(layer1->GetOutputs(), 0);

	//		sf::Event event;
	//		if (window.pollEvent(event))
	//		{
	//			switch (event.type)
	//			{
	//			case sf::Event::Closed:
	//				exit(0);
	//				break;

	//			case sf::Event::KeyPressed:
	//				if (event.key.code == sf::Keyboard::Escape)
	//				{
	//					exit(0);
	//				}
	//				break;
	//			case sf::Event::Resized:
	//			{
	//				const sf::Vector2u windowSize = window.getSize();
	//				glViewport(0, 0, windowSize.x, windowSize.y);
	//				break;
	//			}
	//			}
	//		}

	//		window.clear();

	//		sf::Shader::bind(&visualizer);

	//		glUniform2ui(0, inputWidth, inputHeight);
	//		glUniform2ui(1, 8, 4);
	//		glUniform2f(2, 1.f, 0.f);
	//		layer3->GetOutputs().Bind(0);

	//		glBindVertexArray(vao);
	//		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	//		glBindVertexArray(0);

	//		window.display();
	//	}
	//}

	size_t minibatch = 0;
	int mode = 0;

	while (window.isOpen())
	{
		sf::Event event;

		while (window.pollEvent(event))
		{
			switch (event.type)
			{
			case sf::Event::KeyPressed:
				switch (event.key.code)
				{
				case sf::Keyboard::Escape:
					window.close();
					break;

				default:
					if (sf::Keyboard::Num1 <= event.key.code && event.key.code <= sf::Keyboard::Num9)
					{
						mode = event.key.code - sf::Keyboard::Num1;
					}
					break;
				}
				break;

			case sf::Event::Closed:
				window.close();
				break;

			case sf::Event::Resized:
			{
				const sf::Vector2u windowSize = window.getSize();
				glViewport(0, 0, windowSize.x, windowSize.y);
				break;
			}
			}
		}

		const GLsizeiptr inputsOffset = (GLsizeiptr)minibatch * inputMinibatchSize;

		spatialPooler->Evaluate(inputs, inputsOffset);
		linear->Evaluate(spatialPooler->GetOutputs(), 0);
		//sigmoid->Evaluate(linear->GetOutputs());
		
		mse->EvaluateGradients(inputs, inputsOffset, linear->GetOutputs(), linear->GetGradients());
		//sigmoid->EvaluateGradients(linear->GetOutputs(), linear->GetGradients());
		linear->EvaluateGradients(spatialPooler->GetGradients());
		spatialPooler->EvaluateGradients();
		
		linear->Update(spatialPooler->GetOutputs(), 0);
		spatialPooler->Update(inputs, inputsOffset);
		
		static GLuint iteration = 0;

		if (iteration++ % 100 == 0)
		{
			GLfloat mean = 0.f, sqrMean = 0.f;

			for (GLfloat dutyCycle : spatialPooler->GetDutyCycles().GetData())
			{
				mean += dutyCycle;
				sqrMean += dutyCycle * dutyCycle;
			}

			mean /= spatialPooler->GetTotalMinicolumnCount();
			sqrMean /= spatialPooler->GetTotalMinicolumnCount();

			std::cout << mean << ' ' << sqrMean - mean * mean << std::endl;

			/*float error = 0.f;

			std::vector<GLfloat> outputs = linear->GetOutputs().GetData();

			for (size_t i = 0; i < outputs.size(); i++)
			{
				error += powf(buffer[inputsOffset + i] / 255.f - outputs[i], 2.f);
			}

			std::cout << 0.5f / config.minibatchSize * error << std::endl;*/
		}

		minibatch = (minibatch + 1) % (inputCount / config.minibatchSize);

		window.clear();

		sf::Shader::bind(&visualizer);
		
		int minibatchWidth = 1 << (int)ceil(log2(sqrt(config.minibatchSize)));
		int minibatchHeight = config.minibatchSize / minibatchWidth;

		int minicolumnsWidth = 1 << (int)ceil(log2(sqrt(spatialPooler->GetMinicolumnCount())));
		int minicolumnsHeight = spatialPooler->GetMinicolumnCount() / minicolumnsWidth;

		glUniform1i(3, GL_FALSE);

		switch (mode)
		{
		case 0:
		{
			glUniform2ui(0, inputWidth, inputHeight);
			glUniform2ui(1, minicolumnsWidth, minicolumnsHeight);
			glUniform2f(2, 10.f, 0.f);
			spatialPooler->GetWeights().Bind(0);
			break;
		}

		case 1:
		{
			glUniform2ui(0, minicolumnsWidth, minicolumnsHeight);
			glUniform2ui(1, minibatchWidth, minibatchHeight);
			glUniform2f(2, 50.f, -50.f * spatialPooler->GetSparsity());
			spatialPooler->GetDutyCycles().Bind(0);
			break;
		}

		case 2:
		{
			glUniform2ui(0, minicolumnsWidth, minicolumnsHeight);
			glUniform2ui(1, minibatchWidth, minibatchHeight);
			glUniform2f(2, 1.f, 0.f);
			spatialPooler->GetMinicolumns().Bind(0);
			break;
		}

		case 3:
		{
			glUniform2ui(0, inputWidth, inputHeight);
			glUniform2ui(1, minicolumnsWidth, minicolumnsHeight);
			glUniform2f(2, 10.f, 0.f);
			glUniform1i(3, GL_TRUE);
			linear->GetWeights().Bind(0);
			break;
		}
		
		case 4:
		{
			glUniform2ui(0, inputWidth, inputHeight);
			glUniform2ui(1, 8, 4);
			glUniform2f(2, 1.f, 0.f);
			linear->GetOutputs().Bind(0);
			break;
		}
		}

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);

		window.display();
	}

	if (mse != nullptr)
	{
		delete mse;
	}

	if (sigmoid != nullptr)
	{
		delete sigmoid;
	}

	if (linear != nullptr)
	{
		delete linear;
	}

	if (spatialPooler != nullptr)
	{
		delete spatialPooler;
	}

	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);
	vao = vbo = 0;

	return 0;
}