// main
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

// OpenEngine stuff
#include <Meta/Config.h>
#include <Logging/Logger.h>
#include <Logging/StreamLogger.h>
#include <Core/Engine.h>
#include <Display/Viewport.h>
#include <Renderers/IRenderingView.h>
#include <Math/Vector.h>

// SimpleSetup
#include <Utils/SimpleSetup.h>

// SDL
#include <Display/SDLEnvironment.h>

// MRI
#include <Renderers/OpenGL/MRIRenderingView.h>

#include <Utils/MoveHandler.h>

using namespace OpenEngine::Core;
using namespace OpenEngine::Display;
using namespace OpenEngine::Logging;
using namespace OpenEngine::Math;
using namespace OpenEngine::Renderers;
using namespace OpenEngine::Renderers::OpenGL;
using namespace OpenEngine::Utils;

/**
 * Main.
 */
int main(int argc, char** argv) {
    IEnvironment* env = new SDLEnvironment(1024,768,32);

    IRenderingView* rv = new MRIRenderingView();

    // Create simple setup
    SimpleSetup* setup = new SimpleSetup("SpinOff", env, rv);

    // Create Scene
    MRINode* mrinode = new MRINode();
    setup->GetEngine().ProcessEvent().Attach(*mrinode);
    setup->SetScene(*mrinode);

    // Register the handler as a listener on up and down keyboard events.
    MoveHandler* move_h = new MoveHandler(*(setup->GetCamera()), setup->GetMouse());
    setup->GetKeyboard().KeyEvent().Attach(*move_h);
    setup->GetEngine().InitializeEvent().Attach(*move_h);
    setup->GetEngine().ProcessEvent().Attach(*move_h);
    setup->GetEngine().DeinitializeEvent().Attach(*move_h);

    setup->GetCamera()->SetPosition(Vector<3, float>(20, 20, 0));
    //setup->GetCamera()->SetDirection(Vector<3, float>(-1,-1,0), Vector<3, float>(0,0,1));
    setup->GetCamera()->LookAt(0, 0, 0);

    // Start the engine.
    setup->GetEngine().Start();

    // Return when the engine stops.
    return EXIT_SUCCESS;
}


