//------------------------------------------------------------------------------
//  Stand-alone GLUT Test Program Shell
//------------------------------------------------------------------------------

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <inttypes.h>

#define i64 int64_t
#define u64 uint64_t


GLFWwindow *window = NULL;
bool g_cursorOverWindow = false;
int windowWidth = -1, windowHeight = -1;

struct ShaderLocations {
	GLint colours, normals, occlusions, positions;
	GLint useLighting, useAmbientOcclusion, useSingleColour, passThrough;
	GLint singleColour;
	GLint cameraInverseT, cameraInverseRx, cameraInverseRy, cameraInverseRz;
	GLint perspectiveProjection, modelMatrix;
	GLint lightPosition, lightIntensity;
};

GLuint screate(char  *src, GLenum type)
{
	GLuint shdr = glCreateShader(type);
	glShaderSource(shdr, 1, (const GLchar **) &src, NULL);
	return shdr;
}

GLint scompile(GLuint shdr, char **err)
{
	glCompileShader(shdr);
	GLint stat;
	glGetShaderiv(shdr, GL_COMPILE_STATUS, &stat);
	if (!stat) {
		GLint len;
		glGetShaderiv(shdr, GL_INFO_LOG_LENGTH, &len);
		*err = (char *) malloc(len * sizeof **err);
		glGetShaderInfoLog(shdr, len, &len, *err);
	}
	return stat;
}

i64 flen(FILE *f)
{
	i64 len;

	if (fseek(f, 0, SEEK_END))
		return -1;
	if ((len = ftell(f)) == -1)
		return -1;
	if (fseek(f, 0, SEEK_SET))
		return -1;

	return len;
}

u64 fload(FILE *f, u64 len, char *buf)
{
	if (fread(buf, sizeof *buf, len, f) != len)
		return 0;
	return 1;
}


GLuint loadShader(GLenum type, const char *fileName, char **message)
{
	FILE *file = fopen(fileName, "r");
	if (!file) {
		return 0;
	}
	i64 fileLength = flen(file);
	if (!fileLength == -1) {
		fclose(file);
		return 0;
	}
	char *buf = (char *) calloc(fileLength + 1, sizeof(char));
	if (!buf) {
		fclose(file);
		return 0;
	}
	if (!fload(file, fileLength, buf)) {
		free(buf);
		fclose(file);
		return 0;
	}
	fclose(file);
	GLuint ID = screate(buf, type);
	if (!ID) {
		free(buf);
		return 0;
	}
	free(buf);
	if (!scompile(ID, message)) {
		return 0;
	}
	return ID;
}

typedef float m4f[16];

static void zero(m4f m)
{
	memset(m, 0, 16 * sizeof(m[0]));
}

void pers(m4f m, float n, float f, float l, float r, float t, float b)
{
	zero(m);
	float t0 = (r + l) / (r - l);
	float t1 = (t + b) / (t - b);
	float t2 = -((f + n) / (f - n));
	float t3 = -((2.0f * f * n) / (f - n));
	m[0] = (2.0f * n) / (r - l);
	m[2] = t0;
	m[5] = (2.0f * n) / (t - b);
	m[6] = t1;
	m[10] = t2;
	m[11] = t3;
	m[14] = -1.0f;
}


//-----------------------------------------------------------------------

void checkGLError( const char hdr[] )
{
    int err = glGetError();
    if( err )
    {
        fprintf(stderr, "ERROR %s: %s\n", hdr, gluErrorString(err));
        exit(1);
    }
}


void reshape( int width, int height )
{
    glViewport(0, 0, width, height);
}

//-----------------------------------------------------------------------

void display()
{
    static float angle = 0.0;

    // Clear screen
    int err=0;
    glClearColor( 0.1f, 0.1f, 0.43f, 1.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // Load up PROJECTION and MODELVIEW
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2,2,-2,2,-2,2);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //------------------------------------------------------------
    // Put your GL draw calls here.
    //------------------------------------------------------------

    // Swap
    glutSwapBuffers();

    // Cause display() to be called again.a
    glutPostRedisplay();
    checkGLError( "End of display()" );
}

//-----------------------------------------------------------------------

// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);


int main( int argc, char** argv )
{
    //Init GLFW
    if (!glfwInit()) {
  		fprintf(stderr, "Error: GLFW could not be initialised. Exiting.\n");
  		exit(1);
  	}

  	window = glfwCreateWindow(800, 800, "Renderer", NULL, NULL);
  	if (!window) {
  		printf("Error: A window could not be opened. Exiting.\n");
  		glfwTerminate();
  		exit(1);
  	}
  	windowWidth = 800;
  	windowHeight = 800;

  	glfwMakeContextCurrent(window);
    // Set the required callback functions
    glfwSetKeyCallback(window, key_callback);

  	GLenum result = glewInit();
  	if (result != GLEW_OK) {
  		printf("Error: GLEW could not be initialised.\n");
  		printf("%s. Exiting.\n", glewGetErrorString(result));
  		glfwTerminate();
  		exit(1);
  	}

    printf( "GL_RENDERER = %s\n", glGetString( GL_RENDERER) );

    const char *fshader = "shaders/fragmentShader.glsl";
    const char *vshader = "shaders/vshader.glsl";

    char *shaderMessage = NULL;
  	GLuint fshdr = loadShader(GL_FRAGMENT_SHADER, fshader, &shaderMessage);
  	if (shaderMessage) {
  		fprintf(stderr, "%s\n", shaderMessage);
  		shaderMessage = NULL;
  	}
  	GLuint vshdr = loadShader(GL_VERTEX_SHADER, vshader, &shaderMessage);
  	if (shaderMessage) {
  		fprintf(stderr, "%s\n", shaderMessage);
  		shaderMessage = NULL;
  	}

  // 	/* Link the shaders. */
  // 	GLuint shader = glCreateProgram();
  // 	glAttachShader(shader, fshdr);
  // 	glAttachShader(shader, vshdr);
  // 	glLinkProgram(shader);
  // 	GLint stat;
  // 	glGetProgramiv(shader, GL_LINK_STATUS, &stat);
  // 	if (!stat) {
  // 		GLint len;
  // 		glGetProgramiv(shader, GL_INFO_LOG_LENGTH, &len);
  // 		GLchar err[len];
  // 		glGetProgramInfoLog(shader, len, &len, err);
  // 		printf("%s\n", err);
  // 		glfwTerminate();
  // 		return EXIT_FAILURE;
  // 	}
  // 	glUseProgram(shader);
	//
  // 	struct ShaderLocations locs = {
  // 		.colours = glGetAttribLocation(shader, "colour"),
  // 		.normals = glGetAttribLocation(shader, "normal"),
  // 		.occlusions = glGetAttribLocation(shader, "occlusion"),
  // 		.positions = glGetAttribLocation(shader, "position"),
	//
  // 		.useLighting = glGetUniformLocation(shader, "useLighting"),
  // 		.useAmbientOcclusion = glGetUniformLocation(shader, "useAmbientOcclusion"),
  // 		.useSingleColour = glGetUniformLocation(shader, "useSingleColour"),
  // 		.passThrough = glGetUniformLocation(shader, "passThrough"),
  // 		.singleColour = glGetUniformLocation(shader, "singleColour"),
	//
  // 		.cameraInverseT = glGetUniformLocation(shader, "cameraInverseT"),
  // 		.cameraInverseRx = glGetUniformLocation(shader, "cameraInverseRx"),
  // 		.cameraInverseRy = glGetUniformLocation(shader, "cameraInverseRy"),
  // 		.cameraInverseRz = glGetUniformLocation(shader, "cameraInverseRz"),
  // 		.perspectiveProjection = glGetUniformLocation(shader, "perspectiveProjection"),
  // 		.modelMatrix = glGetUniformLocation(shader, "modelMatrix"),
	//
  // 		.lightPosition = glGetUniformLocation(shader, "lightPosition"),
  // 		.lightIntensity = glGetUniformLocation(shader, "lightIntensity")
  // 	};
	//
	//
	//
  // 	/* Load the perspective projection matrix into the shaders. */
  // 	float perspectiveMat[16];
  // 	pers(perspectiveMat, 1.0f, 1000.0f, -1.0f, 1.0f, 1.0f, -1.0f);
  // 	glUniformMatrix4fv(locs.perspectiveProjection, 1, GL_TRUE, perspectiveMat);
	//
  // /* ************************************************************************** */
	//
  // 	/* Light position should be a function of the scene size and oblique
  // 	 * to the voxel faces for good results */
  // 	const float lightX = 64 / 3.1f;
  // 	const float lightY = 2000;
  // 	const float lightZ = 64 / 2.1f;
  // 	const float lightIntensity = 2.5f;
	//
  // 	glUniform4f(locs.lightPosition, lightX, lightY, lightZ, 1.0f);
  // 	glUniform1f(locs.lightIntensity, lightIntensity);




    // Set up vertex data (and buffer(s)) and attribute pointers
    GLfloat vertices[] = {
        -0.5f, -0.5f, 0.0f, // Left
         0.5f, -0.5f, 0.0f, // Right
         0.0f,  0.5f, 0.0f  // Top
    };
    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    // Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0); // Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind

    glBindVertexArray(0); // Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs)


    // Game loop
    while (!glfwWindowShouldClose(window))
    {
        // Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
        glfwPollEvents();

        // Render
        // Clear the colorbuffer
        glClearColor(0.6f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw our first triangle
        //glUseProgram(shader);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);

        // Swap the screen buffers
        glfwSwapBuffers(window);
    }
    // Properly de-allocate all resources once they've outlived their purpose
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    // Terminate GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();
    return 0;

}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

}
