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
using namespace OpenEngine::Renderers;
using namespace OpenEngine::Renderers::OpenGL;
using namespace OpenEngine::Utils;

/**
 * Main.
 */
int main(int argc, char** argv) {
    IEnvironment* env = new SDLEnvironment(1024,768,32);

    //Viewport* vp = new Viewport(env->GetFrame());

    IRenderingView* rv = new MRIRenderingView();

    // Create simple setup
    SimpleSetup* setup = new SimpleSetup("SpinOff", vp, env, rv);

    // Create Scene
    ISceneNode* mrinode = new MRINode();
    setup->SetScene(*mrinode);

    // Add cam
    /*
    ToolChain* chain = new ToolChain();
    CameraTool* ct = new CameraTool(false);
    chain->PushBackTool(ct);    

    MouseSelection* ms = new MouseSelection(env->GetFrame(), setup->GetMouse(), NULL);
    ms->BindTool(vp, chain);
    
    setup->GetKeyboard().KeyEvent().Attach(*ms);
    setup->GetMouse().MouseMovedEvent().Attach(*ms);
    setup->GetMouse().MouseButtonEvent().Attach(*ms);
    setup->GetRenderer().PostProcessEvent().Attach(*ms);
    */
    
    // Register the handler as a listener on up and down keyboard events.
    MoveHandler* move_h = new MoveHandler(*(setup->GetCamera()), setup->GetMouse());
    setup->GetKeyboard()->KeyEvent().Attach(*move_h);
    setup->GetEngine().InitializeEvent().Attach(*move_h);
    setup->GetEngine().ProcessEvent().Attach(*move_h);
    setup->GetEngine().DeinitializeEvent().Attach(*move_h);


    // Start the engine.
    setup->GetEngine().Start();

    // Return when the engine stops.
    return EXIT_SUCCESS;
}


