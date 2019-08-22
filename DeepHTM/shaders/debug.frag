#version 430

layout(location = 0) uniform uvec2 size;

layout(binding = 0) buffer Data
{
	float data[];
};

in vec2 coord;

out vec4 color;

void main()
{
	color = vec4(data[uint(coord.x * (size.x - 1u)) + uint(coord.y * (size.y - 1u)) * size.x].xxx, 1.0);
}