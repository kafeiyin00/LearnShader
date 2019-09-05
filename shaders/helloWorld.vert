#version 400

layout (std140) uniform Matrices {
	mat4 pvm;
} ;

in vec4 position;
//layout(location = 0) in vec3 vp;

out vec4 color;
out vec4 color1;

void main()
{
	color = position;
	color1 = position*2;
	gl_Position = pvm * position ;
	//gl_Position = vec4(vp.x,vp.y,vp.z,1);
	//color = gl_Position;
} 
