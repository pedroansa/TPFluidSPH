// ----------------------------------------------------------------------------
// main.cpp
//
//  Created on: Fri Jan 22 20:45:07 2021
//      Author: Kiwon Um
//        Mail: kiwon.um@telecom-paris.fr
//
// Description: SPH simulator (DO NOT DISTRIBUTE!)
//
// Copyright 2021-2023 Kiwon Um
//
// The copyright to the computer program(s) herein is the property of Kiwon Um,
// Telecom Paris, France. The program(s) may be used and/or copied only with
// the written permission of Kiwon Um or in accordance with the terms and
// conditions stipulated in the agreement/contract under which the program(s)
// have been supplied.
// ----------------------------------------------------------------------------

#define _USE_MATH_DEFINES

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <tuple>

#ifndef M_PI
#define M_PI 3.141592
#endif

#include "Vector.hpp"

// window parameters
GLFWwindow *gWindow = nullptr;
int gWindowWidth = 1024;
int gWindowHeight = 768;

// timer
float gAppTimer = 0.0;
float gAppTimerLastClockTime;
bool gAppTimerStoppedP = true;

// global options
bool gPause = true;
bool gSaveFile = false;
bool gShowGrid = true;
bool gShowVel = false;
int gSavedCnt = 0;

const int kViewScale = 15;

// SPH Kernel function: cubic spline
class CubicSpline {
public:
  explicit CubicSpline(const Real h=1) : _dim(2)
  {
    setSmoothingLen(h);
  }
  void setSmoothingLen(const Real h)
  {
    const Real h2 = square(h), h3 = h2*h;
    _h = h;
    _sr = 2e0*h;
    _c[0]  = 2e0/(3e0*h);
    _c[1]  = 10e0/(7e0*M_PI*h2);
    _c[2]  = 1e0/(M_PI*h3);
    _gc[0] = _c[0]/h;
    _gc[1] = _c[1]/h;
    _gc[2] = _c[2]/h;
  }
  Real smoothingLen() const { return _h; }
  Real supportRadius() const { return _sr; }

  Real f(const Real l) const
  {
    const Real q = l/_h;
    if(q<1e0) return _c[_dim-1]*(1e0 - 1.5*square(q) + 0.75*cube(q));
    else if(q<2e0) return _c[_dim-1]*(0.25*cube(2e0-q));
    return 0;
  }
  Real derivative_f(const Real l) const
  {
    const Real q = l/_h;
    if(q<=1e0) return _gc[_dim-1]*(-3e0*q+2.25*square(q));
    else if(q<2e0) return -_gc[_dim-1]*0.75*square(2e0-q);
    return 0;
  }

  Real w(const Vec2f &rij) const { return f(rij.length()); }
  Vec2f grad_w(const Vec2f &rij) const { return grad_w(rij, rij.length()); }
  Vec2f grad_w(const Vec2f &rij, const Real len) const
  {
    return derivative_f(len)*rij/len;
  }

private:
  unsigned int _dim;
  Real _h, _sr, _c[3], _gc[3];
};

class SphSolver {
public:
  explicit SphSolver(
    const Real nu=0.8, const Real h=0.5, const Real density=1e3,
    const Vec2f g=Vec2f(0, -9.8), const Real eta=0.01, const Real gamma=7.0) :
    _kernel(h), _nu(nu), _h(h), _d0(density),
    _g(g), _eta(eta), _gamma(gamma)
  {
    _dt = 0.0005;
    _m0 = _d0*_h*_h;
    _c = std::fabs(_g.y)/_eta;
    _k = _d0*_c*_c/_gamma;
  }

  // assume an arbitrary grid with the size of res_x*res_y; a fluid mass fill up
  // the size of f_width, f_height; each cell is sampled with 2x2 particles.
  void initScene(
    const int res_x, const int res_y, const int f_width, const int f_height)
  {
    _pos.clear();

    _resX = res_x;
    _resY = res_y;

    // set wall for boundary
    _l = 0.5*_h;
    _r = static_cast<Real>(res_x) - 0.5*_h;
    _b = 0.5*_h;
    _t = static_cast<Real>(res_y) - 0.5*_h;

    // Create the wall with particles
    int count = 0;

    for (int i = 0; i < _r; ++i) {
        _pos.push_back(Vec2f(i + 0.25, _b));
        _pos.push_back(Vec2f(i + 0.75, _b));

        _pos.push_back(Vec2f(i + 0.25, _t));
        _pos.push_back(Vec2f(i + 0.75, _t));

        count += 4;

    }

    for (int i = 0; i < _t; ++i) {
        _pos.push_back(Vec2f(_r, i + 0.25));
        _pos.push_back(Vec2f(_r, i + 0.75));

        _pos.push_back(Vec2f(_l, i + 0.25));
        _pos.push_back(Vec2f(_l, i + 0.75));

        count += 4;

    }

    int n = 2;
    //sample a fluid mass
    for (int j = n; j < f_height + n; ++j) {
        for (int i = n; i < f_width + n; ++i) {
            _pos.push_back(Vec2f(i + 0.25, j + 0.25));
            _pos.push_back(Vec2f(i + 0.75, j + 0.25));
            _pos.push_back(Vec2f(i + 0.25, j + 0.75));
            _pos.push_back(Vec2f(i + 0.75, j + 0.75));
        }
    }

    // make sure for the other particle quantities
    _vel = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _f = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _p   = std::vector<Real>(_pos.size(), 0);
    _d   = std::vector<Real>(_pos.size(), 0);


    _col = std::vector<float>(_pos.size()*4, 1.0); // RGBA
    _vln = std::vector<float>(_pos.size()*4, 0.0); // GL_LINES

    _pidxInGrid = std::vector<std::vector<tIndex>>(_resX * _resY, std::vector<tIndex>());

    _bo = std::vector<bool>(_pos.size(), false);

    // 
    for (int i = 0; i < count; i++) {
        _bo[i] = true;
    }

    updateColor();
  }

  void update()
  {
    std::cout << '.' << std::flush;

    buildNeighbor();
    computeDensity();
    computePressure();

    // Reset force and acceleration
    _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _f = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    applyBodyForce();
    applyPressureForce();
    applyViscousForce();
    updateAcceleration();
    updateVelocity();
    updatePosition();

    //resolveCollision();

    updateColor();
    if(gShowVel) updateVelLine();
  }

  tIndex particleCount() const { return _pos.size(); }
  const Vec2f& position(const tIndex i) const { return _pos[i]; }
  const float& color(const tIndex i) const { return _col[i]; }
  const float& vline(const tIndex i) const { return _vln[i]; }

  int resX() const { return _resX; }
  int resY() const { return _resY; }

  Real equationOfState(
    const Real d, const Real d0,
    const Real k,               // NOTE: You can use _k for k here.
    const Real gamma=7.0)
  {
    // TODO: pressure calculation
  }

private:

    // In these part, for every method, i've just implemented the exact equation of the the assignment
    void buildNeighbor()
    {
        // TODO:
        for (auto& pos : _pidxInGrid) {
            pos.clear();
        }

        for (tIndex i = 0; i < particleCount(); i++) {
            tIndex pos = idx1d(floor(_pos[i].x), floor(_pos[i].y));
            _pidxInGrid[pos].push_back(i);
        }
    }

    std::vector<std::tuple<int, int>> getNeighbors(int pos)
    {
        // TODO:
        // Return the position of the cells arround the position given with the center cell included, and dealing with the boundary
        std::vector<std::tuple<int, int>> neigborhood;
        int x = floor(_pos[pos].x);
        int y = floor(_pos[pos].y);

        neigborhood.push_back(std::make_tuple(x, y));

        if (x > 0) {
            neigborhood.push_back(std::make_tuple(x - 1, y));

            if (y > 0) {
                neigborhood.push_back(std::make_tuple(x - 1, y - 1));
            }
            if (y < _resY - 1) {
                neigborhood.push_back(std::make_tuple(x - 1, y + 1));
            }

        }

        if (x < _resX - 1) {
            neigborhood.push_back(std::make_tuple(x + 1, y));

            if (y > 0) {
                neigborhood.push_back(std::make_tuple(x + 1, y - 1));
            }
            if (y < _resY - 1) {
                neigborhood.push_back(std::make_tuple(x + 1, y + 1));
            }

        }

        if (y > 0) {
            neigborhood.push_back(std::make_tuple(x, y - 1));
        }

        if (y < _resY - 1) {
            neigborhood.push_back(std::make_tuple(x, y + 1));
        }

        return neigborhood;
    }

    void computeDensity()
    {
        // TODO:

        for (tIndex i = 0; i < _d.size(); i++) {
            _d[i] = 0.0f;

            std::vector<std::tuple<int, int>> neighbors = getNeighbors(i);
         
            for (auto &point : neighbors) {
                for (auto &particle : _pidxInGrid[idx1d(std::get<0>(point), std::get<1>(point))]) {
                    auto aux = _pos[i] - _pos[particle];
                    _d[i] += _m0 * _kernel.w(aux);
                }
            }

        }

    }

    void computePressure()
    {
        for (tIndex i = 0; i < _p.size(); i++) {
            _p[i] = std::max(0.0, _k * ((pow(_d[i] / _d0, 7)) - 1));
        }
    }

    void applyBodyForce()
    {
        // TODO:
        for (tIndex i = 0; i < _acc.size(); i++) {
            _f[i] += _m0 * _g;
        }
    }

    void applyPressureForce()
    {
        // TODO:


        for (tIndex i = 0; i < _p.size(); i++) {
            std::vector<std::tuple<int, int>> neighbors = getNeighbors(i);
            Vec2f aux = Vec2f(0,0);

            for (auto &point : neighbors) {
                for (auto &particle : _pidxInGrid[idx1d(std::get<0>(point), std::get<1>(point))]) {
                    if (particle != i) {

                        aux += _m0 * _kernel.grad_w(_pos[i] - _pos[particle]) * (_p[i] / _d[i] / _d[i] + _p[particle] / _d[particle] / _d[particle]);
                    }

                }
            }

            _f[i] -= _m0 * aux;

        }
    }

    void applyViscousForce()
    {
        // TODO:
        for (tIndex i = 0; i < _pos.size(); i++) {
            std::vector<std::tuple<int, int>> neighbors = getNeighbors(i);
            Vec2f aux = Vec2f(0, 0);

            for (auto &point : neighbors) {
                for (auto &particle : _pidxInGrid[idx1d(std::get<0>(point), std::get<1>(point))]) {
                    if (particle != i) {
                        aux +=  _m0  / _d[particle] * (_vel[i] - _vel[particle]) * _kernel.grad_w(_pos[i] - _pos[particle]) * (_pos[i] - _pos[particle]) / ((_pos[i] - _pos[particle]) * (_pos[i] - _pos[particle]) + 0.01 * _h * _h);
                    }

                }
            }

            _f[i] += (2 * _nu * _m0) * aux;

        }


    }

    void updateAcceleration()
    {
        // TODO:
    // Calculate the new acceleration based on the new force
        for (int i = 0; i < _acc.size(); i++) {
            // a = f / m
            _acc[i] = _f[i] / _m0;
        }
    }

    void updateVelocity()
    {
        // TODO:
        // Here we make sure that the boundaries particles won't get any velocity and won't move 
        for (tIndex i = 0; i < _vel.size(); i++) {
            if (_bo[i] == true) {
                _vel[i] = Vec2f(0,0);
                continue;
            }
            _vel[i] += _dt * _acc[i];
        }
    }

    void updatePosition()
    {
        // TODO:
        for (tIndex i = 0; i < _pos.size(); i++) {
            _pos[i] += _dt * _vel[i];
        }
    }

  // simple collision detection/resolution for each particle
  void resolveCollision()
  {
    std::vector<tIndex> need_res;
    for(tIndex i=0; i<particleCount(); ++i) {
      if(_pos[i].x<_l || _pos[i].y<_b || _pos[i].x>_r || _pos[i].y>_t)
        need_res.push_back(i);
    }

    for(
      std::vector<tIndex>::const_iterator it=need_res.begin();
      it<need_res.end();
      ++it) {
      const Vec2f p0 = _pos[*it];
      _pos[*it].x = clamp(_pos[*it].x, _l, _r);
      _pos[*it].y = clamp(_pos[*it].y, _b, _t);
      _vel[*it] = (_pos[*it] - p0)/_dt;
    }
  }

  void updateColor()
  {
    for(tIndex i=0; i<particleCount(); ++i) {
      _col[i*4+0] = 0.6;
      _col[i*4+1] = 0.6;
      _col[i*4+2] = _d[i]/_d0;
    }
  }

  void updateVelLine()
  {
    for(tIndex i=0; i<particleCount(); ++i) {
      _vln[i*4+0] = _pos[i].x;
      _vln[i*4+1] = _pos[i].y;
      _vln[i*4+2] = _pos[i].x + _vel[i].x;
      _vln[i*4+3] = _pos[i].y + _vel[i].y;
    }
  }

  inline tIndex idx1d(const int i, const int j) { return i + j*resX(); }

  const CubicSpline _kernel;

  // particle data
  std::vector<bool> _bo; //
  std::vector<Vec2f> _pos;      // position
  std::vector<Vec2f> _vel;      // velocity
  std::vector<Vec2f> _acc;      // acceleration
  std::vector<Real>  _p;        // pressure
  std::vector<Real>  _d;        // density
  std::vector<Vec2f>  _f;        // force

  std::vector< std::vector<tIndex> > _pidxInGrid; // will help you find neighbor particles

  std::vector<float> _col;    // particle color; just for visualization
  std::vector<float> _vln;    // particle velocity lines; just for visualization

  // simulation
  Real _dt;                     // time step

  int _resX, _resY;             // background grid resolution

  // wall
  Real _l, _r, _b, _t;          // wall (boundary)

  // SPH coefficients
  Real _nu;                     // viscosity coefficient
  Real _d0;                     // rest density
  Real _h;                      // particle spacing (i.e., diameter)
  Vec2f _g;                     // gravity

  Real _m0;                     // rest mass
  Real _k;                      // EOS coefficient

  Real _eta;
  Real _c;                      // speed of sound
  Real _gamma;                  // EOS power factor
};

SphSolver gSolver(0.08, 0.5, 1e3, Vec2f(0, -9.8), 0.01, 7.0);

void printHelp()
{
  std::cout <<
    "> Help:" << std::endl <<
    "    Keyboard commands:" << std::endl <<
    "    * H: print this help" << std::endl <<
    "    * P: toggle simulation" << std::endl <<
    "    * G: toggle grid rendering" << std::endl <<
    "    * V: toggle velocity rendering" << std::endl <<
    "    * S: save current frame into a file" << std::endl <<
    "    * Q: quit the program" << std::endl;
}

// Executed each time the window is resized. Adjust the aspect ratio and the rendering viewport to the current window.
void windowSizeCallback(GLFWwindow *window, int width, int height)
{
  gWindowWidth = width;
  gWindowHeight = height;
  glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

// Executed each time a key is entered.
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if(action == GLFW_PRESS && key == GLFW_KEY_H) {
    printHelp();
  } else if(action == GLFW_PRESS && key == GLFW_KEY_S) {
    gSaveFile = !gSaveFile;
  } else if(action == GLFW_PRESS && key == GLFW_KEY_G) {
    gShowGrid = !gShowGrid;
  } else if(action == GLFW_PRESS && key == GLFW_KEY_V) {
    gShowVel = !gShowVel;
  } else if(action == GLFW_PRESS && key == GLFW_KEY_P) {
    gAppTimerStoppedP = !gAppTimerStoppedP;
    if(!gAppTimerStoppedP)
      gAppTimerLastClockTime = static_cast<float>(glfwGetTime());
  } else if(action == GLFW_PRESS && key == GLFW_KEY_Q) {
    glfwSetWindowShouldClose(window, true);
  }
}

void initGLFW()
{
  // Initialize GLFW, the library responsible for window management
  if(!glfwInit()) {
    std::cerr << "ERROR: Failed to init GLFW" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Before creating the window, set some option flags
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // only if requesting 3.0 or above
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE); // for OpenGL below 3.2
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

  // Create the window
  gWindowWidth = gSolver.resX()*kViewScale;
  gWindowHeight = gSolver.resY()*kViewScale;
  gWindow = glfwCreateWindow(
    gSolver.resX()*kViewScale, gSolver.resY()*kViewScale,
    "Basic SPH Simulator", nullptr, nullptr);
  if(!gWindow) {
    std::cerr << "ERROR: Failed to open window" << std::endl;
    glfwTerminate();
    std::exit(EXIT_FAILURE);
  }

  // Load the OpenGL context in the GLFW window
  glfwMakeContextCurrent(gWindow);

  // not mandatory for all, but MacOS X
  glfwGetFramebufferSize(gWindow, &gWindowWidth, &gWindowHeight);

  // Connect the callbacks for interactive control
  glfwSetWindowSizeCallback(gWindow, windowSizeCallback);
  glfwSetKeyCallback(gWindow, keyCallback);

  std::cout << "Window created: " <<
    gWindowWidth << ", " << gWindowHeight << std::endl;
}

void clear();

void exitOnCriticalError(const std::string &message)
{
  std::cerr << "> [Critical error]" << message << std::endl;
  std::cerr << "> [Clearing resources]" << std::endl;
  clear();
  std::cerr << "> [Exit]" << std::endl;
  std::exit(EXIT_FAILURE);
}

void initOpenGL()
{
  // Load extensions for modern OpenGL
  if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    exitOnCriticalError("[Failed to initialize OpenGL context]");

  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);

  glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

void init()
{
  gSolver.initScene(48, 32, 16, 16);

  initGLFW();                   // Windowing system
  initOpenGL();
}

void clear()
{
  glfwDestroyWindow(gWindow);
  glfwTerminate();
}

// The main rendering call
void render()
{
  glClearColor(.4f, .4f, .4f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // grid guides
  if(gShowGrid) {
    glBegin(GL_LINES);
    for(int i=1; i<gSolver.resX(); ++i) {
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(i), 0.0);
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(i), static_cast<Real>(gSolver.resY()));
    }
    for(int j=1; j<gSolver.resY(); ++j) {
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(0.0, static_cast<Real>(j));
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(gSolver.resX()), static_cast<Real>(j));
    }
    glEnd();
  }

  // render particles
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  glPointSize(0.25f*kViewScale);

  glColorPointer(4, GL_FLOAT, 0, &gSolver.color(0));
  glVertexPointer(2, GL_FLOAT, 0, &gSolver.position(0));
  glDrawArrays(GL_POINTS, 0, gSolver.particleCount());

  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  // velocity
  if(gShowVel) {
    glColor4f(0.0f, 0.0f, 0.5f, 0.2f);

    glEnableClientState(GL_VERTEX_ARRAY);

    glVertexPointer(2, GL_FLOAT, 0, &gSolver.vline(0));
    glDrawArrays(GL_LINES, 0, gSolver.particleCount()*2);

    glDisableClientState(GL_VERTEX_ARRAY);
  }

  if(gSaveFile) {
    std::stringstream fpath;
    fpath << "s" << std::setw(4) << std::setfill('0') << gSavedCnt++ << ".tga";

    std::cout << "Saving file " << fpath.str() << " ... " << std::flush;
    const short int w = gWindowWidth;
    const short int h = gWindowHeight;
    std::vector<int> buf(w*h*3, 0);
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, &(buf[0]));

    FILE *out = fopen(fpath.str().c_str(), "wb");
    short TGAhead[] = {0, 2, 0, 0, 0, 0, w, h, 24};
    fwrite(&TGAhead, sizeof(TGAhead), 1, out);
    fwrite(&(buf[0]), 3*w*h, 1, out);
    fclose(out);
    gSaveFile = false;

    std::cout << "Done" << std::endl;
  }
}

// Update any accessible variable based on the current time
void update(const float currentTime)
{
  if(!gAppTimerStoppedP) {
    // NOTE: When you want to use application's dt ...
    const float dt = currentTime - gAppTimerLastClockTime;
    gAppTimerLastClockTime = currentTime;
    gAppTimer += dt;

    // solve 10 steps
    gSolver.update();
  }
}

int main(int argc, char **argv)
{
  init();
  while(!glfwWindowShouldClose(gWindow)) {
    update(static_cast<float>(glfwGetTime()));
    render();
    glfwSwapBuffers(gWindow);
    glfwPollEvents();
  }
  clear();
  std::cout << " > Quit" << std::endl;
  return EXIT_SUCCESS;
}
