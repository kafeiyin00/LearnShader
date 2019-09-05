#version 400

in vec4 color1;
in vec4 color;

layout(location = 0) out vec4 outputF;

void main()
{
	outputF = vec4(1.0, 0.5, 0.25, 1.0);
	outputF = color1;
} 
