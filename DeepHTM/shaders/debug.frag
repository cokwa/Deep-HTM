#version 430

layout(location = 0) uniform uvec2 size;
layout(location = 1) uniform uvec2 nums;

layout(location = 2) uniform vec2 coeff;

layout(location = 3) uniform bool insideOut = false;

layout(binding = 0) buffer Data
{
	float data[];
};

in vec2 coord;

out vec4 color;

void main()
{
	uvec2 total = size * nums;
	uvec2 local = uvec2(coord.x * total.x, (1.0 - coord.y) * total.y) % size, group = uvec2(coord.x * nums.x, (1.0 - coord.y) * nums.y);
	float dat = coeff.x * data[(local.x + local.y * size.x) * (insideOut ? nums.x * nums.y : 1) + (group.x + group.y * nums.x) * (insideOut ? 1 : size.x * size.y)] + coeff.y;

	color = vec4(vec3(clamp(dat, 0.0, 1.0), 0.0, clamp(-dat, 0.0, 1.0)), 1.0);
}