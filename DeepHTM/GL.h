#pragma once

#include<glad/glad.h>

#include<vector>
#include<string>
#include<functional>
#include<fstream>
#include<random>

namespace DeepHTM
{
	namespace GL
	{
		class ComputeShader
		{
		private:
			GLint program;

		public:
			ComputeShader(const std::string& path, const std::string& string = "") : program(0)
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

				if (!string.empty())
				{
					const size_t versionLine = source.find('\n') + 1;
					source =
						source.substr(0, versionLine) +
						'\n' + string + '\n' +
						source.substr(versionLine, source.length());
				}

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

					//TODO: ugly
					throw std::runtime_error(sources[0] + std::string("\n:\n") + infoLog);
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

					//TODO: ugly
					throw std::runtime_error(sources[0] + std::string("\n:\n") + infoLog);
				}
			}

			ComputeShader(ComputeShader&& other) : program(other.program)
			{
				other.program = 0;
			}

			virtual ~ComputeShader()
			{
				if (program != 0)
				{
					glDeleteProgram(program);
					program = 0;
				}
			}

			void Use() const
			{
				glUseProgram(program);
			}
		};

		template<class T>
		class ShaderStorageBuffer
		{
		private:
			GLsizeiptr length;
			GLenum usage;

			GLuint ssbo;

		public:
			ShaderStorageBuffer(GLsizeiptr length, const T* data, GLenum usage) : length(length), usage(usage), ssbo(0)
			{
				glGenBuffers(1, &ssbo);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(T), data, usage);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}

			ShaderStorageBuffer(GLsizeiptr length, const T* data) : ShaderStorageBuffer(length, data, GL_DYNAMIC_COPY)
			{

			}

			ShaderStorageBuffer(GLsizeiptr length) : ShaderStorageBuffer(length, nullptr)
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

			GLsizeiptr GetLength() const
			{
				return length;
			}

			GLenum GetUsage() const
			{
				return usage;
			}

			template<class It>
			void GetData(It data) const
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				const T* buffer = reinterpret_cast<const T*>(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY));

				for (GLsizeiptr i = 0; i < length; i++)
				{
					*data++ = buffer[i];
				}

				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}

			std::vector<T> GetData() const
			{
				std::vector<T> data(length);
				GetData(data.begin());

				return data;
			}

			void SetData(const std::function<T()>& settor)
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				T* buffer = reinterpret_cast<T*>(glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY));

				for (GLsizeiptr i = 0; i < length; i++)
				{
					buffer[i] = settor();
				}

				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}

			void SetData(const std::vector<T>& data)
			{
				SetData(data.begin());
			}

			void Bind(GLuint index) const
			{
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, ssbo);
			}

			void Bind(GLuint index, GLintptr offset, GLsizeiptr length) const
			{
				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, index, ssbo, offset * sizeof(T), length * sizeof(T));
			}

			//TODO: generalize and parallelize
			void Randomize(T range = (T)1.0)
			{
				std::default_random_engine generator;
				std::normal_distribution<T> distribution;

				SetData([&]() { return distribution(generator) * range; });
			}
		};
	}
}