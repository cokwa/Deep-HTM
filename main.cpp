#include"DeepHTM.h"

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>

#pragma comment(lib, "sfml-system.lib")
#pragma comment(lib, "sfml-window.lib")
#pragma comment(lib, "sfml-graphics.lib")
#pragma comment(lib, "sfml-audio.lib")
#pragma comment(lib, "sfml-network.lib")

#include<iostream>

int main()
{
	sf::RenderWindow window(sf::VideoMode(640, 480), "DeepHTM");
	
	if (!gladLoadGL())
	{
		throw std::exception();
	}

	sf::RectangleShape rect((sf::Vector2f)window.getSize());
	sf::Shader visualizer;

	std::string vert = \
		"#version 430\n"
		"\n"
		"void main()\n"
		"{\n"
		"	gl_Position = gl_ModelViewProjecctionMatrix * gl_Vertex;\n"
		"	gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;\n"
		"	gl_FontColor = gl_Color;\n"
		"}\n";

	std::string frag = \
		"#version 430\n"
		"\n"
		"void main()\n"
		"{\n"
		"	gl_FragColor = gl_Color;\n"
		"}\n";

	std::cout << vert << std::endl << frag << std::endl;

	sf::MemoryInputStream vertStream, fragStream;
	vertStream.open(vert.c_str(), vert.length());
	fragStream.open(frag.c_str(), frag.length());
	visualizer.loadFromStream(vertStream, fragStream);

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
			if (event.type == sf::Event::Closed)
			{
				window.close();
			}
		}

		window.clear();
		window.draw(rect);
		window.display();
	}

	if (deepHTM != nullptr)
	{
		delete deepHTM;
	}

	return 0;
}