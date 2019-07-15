#pragma once

#include<glad/glad.h>

#include<string>
#include<fstream>

namespace DeepHTM
{
	class ComputeShader
	{
	private:
		GLint program;

	public:
		ComputeShader(const std::string& path) : program(0)
		{
			std::ifstream file(path);
			if (!file)
			{
				throw std::runtime_error(path);
			}

			file.seekg(0, file.end);
			std::string source(file.tellg(), '\0');
			file.seekg(0, file.beg);
			file.read(&source[0], source.length());
			file.close();

			GLint shader = glCreateShader(GL_COMPUTE_SHADER);
			const char* sources[]{ source.c_str() };
			glShaderSource(shader, 1, sources, nullptr);
			glCompileShader(shader);

			GLint status;
			glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

			if (status == GL_FALSE)
			{
				GLint length;
				glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

				std::string infoLog(length, '\0');
				glGetShaderInfoLog(shader, length, nullptr, &infoLog[0]);

				throw std::runtime_error(infoLog);
			}

			program = glCreateProgram();
			glAttachShader(program, shader);
			glLinkProgram(program);
			glDetachShader(program, shader);
			glDeleteShader(shader);

			glGetProgramiv(program, GL_LINK_STATUS, &status);

			if (status == GL_FALSE)
			{
				GLint length;
				glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);

				std::string infoLog(length, '\0');
				glGetProgramInfoLog(program, length, nullptr, &infoLog[0]);

				throw std::runtime_error(infoLog);
			}
		}

		virtual ~ComputeShader()
		{
			if (program != 0)
			{
				glDeleteProgram(program);
				program = 0;
			}
		}
	};

	class DeepHTM
	{
	public:
		DeepHTM()
		{
			ComputeShader* computeShader = new ComputeShader("shaders/sp_fully_connected.comp");
			delete computeShader;

			computeShader = new ComputeShader("shaders/sp_k_winner.comp");
			delete computeShader;
		}

		virtual ~DeepHTM()
		{

		}
	};
}