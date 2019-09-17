//#version 330
//
//layout(triangles) in;
//layout(triangles) out;
//
//void main()
//{
//    gl_Position = gl_in[0].gl_Position;
//
//    EmitVertex();
//
//    //gl_Position = gl_in[0].gl_Position + vec4(0.1, 0.0, 0.0, 0.0);
//    //EmitVertex();
//
//    EndPrimitive();
//}

//#version 400
//precision highp float;
//in vec3 vFragColorVs[];
//out vec3 vFragColor;
//
//layout(triangles) in;
//layout(triangle_strip, max_vertices = 3) out;
//
//void main() {
//  for(int i = 0; i < 3; i++) { // for each triangle emit three vertices
//    vFragColor = vFragColorVs[i];
//    vFragColor = vec3(1,0,0);
//    gl_Position = gl_in[i].gl_Position;
//    EmitVertex();
//  }
//  EndPrimitive();
//}


//#version 330
//
//layout(triangles) in;
//layout(triangle_strip, max_vertices = 3) out;
//
//in vData
//{
//    vec3 normal;
//    vec3 color;
//} vertices[];
//
//out vec3 normal;
//out vec3 color;
//   
//
//void main()
//{
//    for(int i = 0;i < 3;i++)
//    {
//        normal = vertices[i].normal;
//        normal = vertices[i].color;
//        color = vec3(1.0,0.0,0.0);
//        gl_Position = gl_in[i].gl_Position;
//        EmitVertex();
//    }
//    EndPrimitive();
//}

#version 330

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

//in vec4 frag_color[];
out vec3 frag_color;

void main()
{	
  int i;
  for(i=0; i<3; i++)
  {
    gl_Position = gl_in[i].gl_Position;
    frag_color = vec3(0.0, 1.0, 0.0);
    frag_color = gl_in[0].gl_Position.xyz;
    EmitVertex();
  }
  EndPrimitive();
}
