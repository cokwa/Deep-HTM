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
	sf::RenderWindow window(sf::VideoMode(640, 480), "DeepHTM");
	
	if (!gladLoadGL())
	{
		throw std::exception();
	}

	sf::Shader visualizer;
	visualizer.loadFromFile("shaders/debug.vert", "shaders/debug.frag");
	sf::Shader::bind(&visualizer);

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

	DeepHTM::DeepHTM* deepHTM = nullptr;

	try
	{
		deepHTM = new DeepHTM::DeepHTM();
	}
	catch (const std::exception& exception)
	{
		std::cout << exception.what() << std::endl;
	}

	while (window.isOpen())
	{
		sf::Event event;

		while (window.pollEvent(event))
		{
			switch (event.type)
			{
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

		window.clear();
		
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);

		window.display();
	}

	if (deepHTM != nullptr)
	{
		delete deepHTM;
	}

	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);
	vao = vbo = 0;

	return 0;
}