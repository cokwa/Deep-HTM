#pragma once

#include<glad/glad.h>

#include<string>
#include<fstream>

namespace DeepHTM
{
	namespace GL
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

		template<class T>
		class ShaderStorageBuffer
		{
		private:
			GLuint ssbo;

		public:
			ShaderStorageBuffer(GLsizeiptr size, const T* data, GLenum usage) : ssbo(0)
			{
				glGenBuffers(1, &ssbo);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, usage);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}

			ShaderStorageBuffer(GLsizeiptr size, const T* data) : ShaderStorageBuffer(size, data, GL_DYNAMIC_COPY)
			{

			}

			ShaderStorageBuffer(GLsizeiptr size) : ShaderStorageBuffer(size, nullptr)
			{

			}

			virtual ~ShaderStorageBuffer()
			{
				if (ssbo != 0)
				{
					glDeleteBuffers(1, &ssbo);
					ssbo = 0;
				}
			}

			void Bind(GLuint index)
			{
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, ssbo);
			}
		};
	}
}