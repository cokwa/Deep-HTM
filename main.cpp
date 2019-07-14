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

	try
	{
		DeepHTM::DeepHTM* deepHTM = new DeepHTM::DeepHTM();
		delete deepHTM;
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
		window.display();
	}

	return 0;
}