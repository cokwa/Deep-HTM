#version 430

layout(location = 0) in vec4 _pos;
layout(location = 1) in vec2 _coord;

out vec2 coord;

void main()
{
	gl_Position = _pos;
	coord = _coord;
}