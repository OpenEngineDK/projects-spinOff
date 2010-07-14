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

#include <Scene/SceneNode.h>

#include <Core/Engine.h>

#include <Display/Viewport.h>
#include <Display/SDLEnvironment.h>
#include <Display/AntTweakBar.h>

#include <Renderers/IRenderingView.h>
#include <Renderers/OpenGL/MRIRenderingView.h>
#include <Renderers/TextureLoader.h>

#include <Math/Vector.h>

#include <Utils/SimpleSetup.h>
#include <Utils/MoveHandler.h>
#include <Utils/CairoTextTool.h>

#include <Utils/InspectionBar.h>

#include "Science/MRIModule.h"

#include <Resources/CairoResource.h>
#include <Resources/DirectoryManager.h>
#include <Resources/ResourceManager.h>


using namespace OpenEngine::Core;
using namespace OpenEngine::Display;
using namespace OpenEngine::Logging;
using namespace OpenEngine::Math;
using namespace OpenEngine::Renderers;
using namespace OpenEngine::Renderers::OpenGL;
using namespace OpenEngine::Utils;
using namespace OpenEngine::Science;
using namespace OpenEngine::Scene;

class MRIHandler : public IListener<KeyboardEventArg> {
private:
    MRINode* node;
public:
    MRIHandler(MRINode* n) { node = n; }

    void Handle(KeyboardEventArg arg){
        if (arg.type == EVENT_PRESS && arg.sym == KEY_f)
            node->Flip(90);
    }
};


// Helpers
static TransformationNode* CreateTextureBillboard(ITextureResourcePtr texture,
                                                  float scale) {
    unsigned int textureHosisontalSize = texture->GetWidth();
    unsigned int textureVerticalSize = texture->GetHeight();

    logger.info << "w x h = " << texture->GetWidth()
                << " x " << texture->GetHeight() << logger.end;
    float fullxtexcoord = 1;
    float fullytexcoord = 1;
  
    FaceSet* faces = new FaceSet();

    float horisontalhalfsize = textureHosisontalSize * 0.5;
    Vector<3,float>* lowerleft = new Vector<3,float>(horisontalhalfsize,0,0);
    Vector<3,float>* lowerright = new Vector<3,float>(-horisontalhalfsize,0,0);
    Vector<3,float>* upperleft = new Vector<3,float>(horisontalhalfsize,textureVerticalSize,0);
    Vector<3,float>* upperright = new Vector<3,float>(-horisontalhalfsize,textureVerticalSize,0);

    FacePtr leftside = FacePtr(new Face(*lowerleft,*lowerright,*upperleft));

    /*
      leftside->texc[1] = Vector<2,float>(1,0);
      leftside->texc[0] = Vector<2,float>(0,0);
      leftside->texc[2] = Vector<2,float>(0,1);
    */
    leftside->texc[1] = Vector<2,float>(0,fullytexcoord);
    leftside->texc[0] = Vector<2,float>(fullxtexcoord,fullytexcoord);
    leftside->texc[2] = Vector<2,float>(fullxtexcoord,0);
    leftside->norm[0] = leftside->norm[1] = leftside->norm[2] = Vector<3,float>(0,0,1);
    leftside->CalcHardNorm();
    leftside->Scale(scale);
    faces->Add(leftside);

    FacePtr rightside = FacePtr(new Face(*lowerright,*upperright,*upperleft));
    /*
      rightside->texc[2] = Vector<2,float>(0,1);
      rightside->texc[1] = Vector<2,float>(1,1);
      rightside->texc[0] = Vector<2,float>(1,0);
    */
    rightside->texc[2] = Vector<2,float>(fullxtexcoord,0);
    rightside->texc[1] = Vector<2,float>(0,0);
    rightside->texc[0] = Vector<2,float>(0,fullytexcoord);
    rightside->norm[0] = rightside->norm[1] = rightside->norm[2] = Vector<3,float>(0,0,1);
    rightside->CalcHardNorm();
    rightside->Scale(scale);
    faces->Add(rightside);

    MaterialPtr m = leftside->mat = rightside->mat = MaterialPtr(new Material());
    m->AddTexture(texture);

    GeometryNode* node = new GeometryNode();
    node->SetFaceSet(faces);
    TransformationNode* tnode = new TransformationNode();
    tnode->AddNode(node);
    return tnode;
}

struct Wall {
    pair<ITextureResourcePtr,string> tex[12];
    TextureLoader& loader;

    Wall(TextureLoader& l) : loader(l) {        
    }

    pair<ITextureResourcePtr,string>& operator()(int x, int y) {
        return tex[x*3+y];
    }
    ISceneNode* MakeScene() {
        SceneNode *sn = new SceneNode();
        CairoTextTool textTool;
        
        for (int x=0;x<4;x++) {
            for (int y=0;y<3;y++) {
                pair<ITextureResourcePtr,string> itm = (*this)(x,y);
                ITextureResourcePtr t = itm.first;
                if (t) {
                    loader.Load(t,TextureLoader::RELOAD_QUEUED);
                    TransformationNode* node = CreateTextureBillboard(t,0.05);
                    node->SetScale(Vector<3,float>(1.0,-1.0,1.0));
                    node->Move(x*35-52,y*25-25,0);

                    CairoResourcePtr textRes = CairoResource::Create(128,32);
                    textRes->Load();

                    ostringstream out;
                    out << "(" << x << "," << y << ") " << itm.second;

                    textTool.DrawText(out.str(), textRes);

                    loader.Load(textRes);
                    TransformationNode* textNode = CreateTextureBillboard(textRes,0.15);
                    textNode->SetScale(Vector<3,float>(1.0,-1.0,1.0));                    
                    textNode->Move(0,23.0,0);


                    node->AddNode(textNode);
                    //sn->AddNode(textNode);
                    sn->AddNode(node);
                }
            }
        }

        return sn;
    }
};



/**
 * Main.
 */
int main(int argc, char** argv) {
    IEnvironment* env = new SDLEnvironment(1024,768,32);

    //IRenderingView* rv = new MRIRenderingView();

    // Create simple setup
    //SimpleSetup* setup = new SimpleSetup("SpinOff", env, rv);
    SimpleSetup* setup = new SimpleSetup("SpinOff", env);
    
    DirectoryManager::AppendPath("./projects/spinOff/report/pics/");

    setup->GetRenderer().SetBackgroundColor(Vector<4,float>(0.0));

    // Ant tweak bar
    AntTweakBar *atb = new AntTweakBar();
    atb->AttachTo(setup->GetRenderer());
    
    setup->GetKeyboard().KeyEvent().Attach(*atb);
    setup->GetMouse().MouseMovedEvent().Attach(*atb);
    setup->GetMouse().MouseButtonEvent().Attach(*atb);


    // Graphics
    ITextureResourcePtr girl = ResourceManager<ITextureResource>::Create("frontpage2.jpg");
    girl->Load();

    // Mri Module
    MRIModule* mri = new MRIModule(girl);
    setup->GetEngine().ProcessEvent().Attach(*mri);
    setup->GetEngine().InitializeEvent().Attach(*mri);
    setup->GetEngine().DeinitializeEvent().Attach(*mri);
    setup->GetKeyboard().KeyEvent().Attach(*mri);

    atb->AddBar(new InspectionBar("mri", Inspect(mri)));

    // Wall
    Wall wall(setup->GetTextureLoader());

    wall(0,0) = make_pair<>(girl, "Girl");
    wall(1,0) = make_pair<>(mri->GetOutputTexture(), "output");
    wall(2,0) = make_pair<>(mri->GetInverseTexture(), "inverse");
    

    ISceneNode *wallNode = wall.MakeScene();
    setup->SetScene(*wallNode);

    float h = -25/2;
    setup->GetCamera()->SetPosition(Vector<3,float>(0.0,h,130));
    setup->GetCamera()->LookAt(Vector<3,float>(0.0,h,0.0));



    // Create Scene
    // MRINode* mrinode = new MRINode();
    // setup->GetEngine().ProcessEvent().Attach(*mrinode);
    // setup->SetScene(*mrinode);

    // MRIHandler handle = MRIHandler(mrinode);
    // setup->GetKeyboard().KeyEvent().Attach(handle);

    // Register the handler as a listener on up and down keyboard events.
    // MoveHandler* move_h = new MoveHandler(*(setup->GetCamera()), setup->GetMouse());
    // setup->GetKeyboard().KeyEvent().Attach(*move_h);
    // setup->GetEngine().InitializeEvent().Attach(*move_h);
    // setup->GetEngine().ProcessEvent().Attach(*move_h);
    // setup->GetEngine().DeinitializeEvent().Attach(*move_h);

    // setup->GetCamera()->SetPosition(Vector<3, float>(20, 20, 0));
    // setup->GetCamera()->LookAt(0, 0, 0);

    // Start the engine.
    setup->GetEngine().Start();    
    
    // Return when the engine stops.
    return EXIT_SUCCESS;
}


