// RL SensOptimizer – BakkesMod Plugin (single-file version)
// Captures in-game inputs & state during Freeplay/Training, computes metrics
// (oversteer, micro-corrections, rotational error, flick responsiveness) and
// prints recommended adjustments for Steering/Aerial sensitivity, Deadzone and
// Dodge Deadzone in the BakkesMod console. No auto-apply.
//
// Build: Use BakkesMod SDK template. Drop this file into the plugin src folder,
// add to project, compile x64 Release. Produces a DLL.
//
// Notes:
// - Reads car & ball state via SDK wrappers.
// - Reads controller input from CarWrapper::GetInput().
// - Runs only in Freeplay/Custom Training.
// - Output only (per user choice I). Does NOT modify your game settings.
//
// © 2025 – Provided "as is"; uses public BakkesMod SDK interfaces.

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <optional>

#include "bakkesmod/plugin/bakkesmodplugin.h"
#include "bakkesmod/wrappers/WrapperStructs.h"
#include "bakkesmod/wrappers/Engine/WorldInfoWrapper.h"
#include "bakkesmod/wrappers/GameEvent/ServerWrapper.h"
#include "bakkesmod/wrappers/GameObject/CarWrapper.h"
#include "bakkesmod/wrappers/GameObject/BallWrapper.h"

// ------------------------------ Helpers ---------------------------------
namespace util {
    inline float clamp(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }
    inline float lerp(float a, float b, float t) { return a + (b - a) * t; }
    inline float length2D(const Vector& v) { return std::sqrt(v.X * v.X + v.Y * v.Y); }
    inline float length3D(const Vector& v) { return std::sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z); }
    inline Vector normalize(const Vector& v) {
        float l = length3D(v);
        if (l <= 1e-6f) return {0,0,0};
        return { v.X / l, v.Y / l, v.Z / l };
    }
    inline float dot2D(const Vector& a, const Vector& b) { return a.X * b.X + a.Y * b.Y; }
    inline float angleDeg2D(const Vector& a, const Vector& b) {
        Vector an = a; an.Z = 0; an = normalize(an);
        Vector bn = b; bn.Z = 0; bn = normalize(bn);
        float d = util::clamp(an.X * bn.X + an.Y * bn.Y + an.Z * bn.Z, -1.0f, 1.0f);
        return std::acos(d) * 180.0f / static_cast<float>(M_PI);
    }
    inline std::string f2(float v, int p=3){ std::ostringstream oss; oss<<std::fixed<<std::setprecision(p)<<v; return oss.str(); }
}

// ------------------------- Metric Aggregator -----------------------------
struct Sample {
    float dt{0.0167f};
    float yaw{0.0f};
    float pitch{0.0f};
    float steer{0.0f};
    float speed{0.0f};
    float angVelYaw{0.0f};
    float rotErrYaw{0.0f};
    bool ballTouched{false};
    float touchSpeed{0.0f};
};

struct Metrics {
    int samples{0};
    float oversteer_sum{0.0f};
    float microcorr_count{0.0f};
    float microcorr_rate{0.0f};
    float rot_err_sum{0.0f};
    float rot_err_max{0.0f};
    float flick_resp_sum{0.0f};
    int   flick_resp_events{0};
    float avg_touch_speed{0.0f};
    int   touches{0};

    void reset(){ *this = Metrics(); }
};

class RLSensOptimizer final : public BakkesMod::Plugin::BakkesModPlugin {
    bool active{false};
    float targetDuration{60.0f};
    float elapsed{0.0f};
    Metrics M;

    float prevYaw{0.0f}, prevPitch{0.0f}, prevSteer{0.0f};
    float prevYawSign{0.0f}, prevPitchSign{0.0f}, prevSteerSign{0.0f};

    float w_oversteer{1.0f};
    float w_micro{1.0f};
    float w_rotctrl{1.0f};
    float w_flick{1.0f};
    float w_balance{1.0f};

public:
    PLUGIN_EXPORT void onLoad() override {
        cvarManager->registerCvar("so_opt_duration", "60", "Seconds to record before computing recommendation", true, true, 10.0f, true, 600.0f)
            .addOnValueChanged([this](std::string, CVarWrapper c){ targetDuration = c.getFloatValue(); });
        cvarManager->registerCvar("so_weights", "1 1 1 1 1", "Weights: oversteer micro rotctrl flick balance", true, false, 0, false, 0)
            .addOnValueChanged([this](std::string, CVarWrapper c){ parseWeights(c.getStringValue()); });

        cvarManager->registerNotifier("so_start", [this](std::vector<std::string>) { start(); }, "Start capture/optimization run", PERMISSION_ALL);
        cvarManager->registerNotifier("so_stop",  [this](std::vector<std::string>) { stopAndReport(); }, "Stop & report recommendation", PERMISSION_ALL);
        cvarManager->registerNotifier("so_status",[this](std::vector<std::string>) { status(); }, "Show status", PERMISSION_ALL);

        gameWrapper->HookEvent("Function TAGame.Car_TA.SetVehicleInput", std::bind(&RLSensOptimizer::OnInput, this, std::placeholders::_1));
        LOG("RL SensOptimizer loaded. Use: so_start / so_stop / so_status");
    }

    PLUGIN_EXPORT void onUnload() override {
        active = false;
    }

private:
    void LOG(const std::string& s){ cvarManager->log("[SensOpt] " + s); }

    void parseWeights(const std::string& s){
        std::istringstream iss(s);
        float a=1,b=1,c=1,d=1,e=1; iss>>a>>b>>c>>d>>e;
        w_oversteer=a; w_micro=b; w_rotctrl=c; w_flick=d; w_balance=e;
    }

    bool inTraining(){
        if (!gameWrapper || !gameWrapper->IsInGame()) return false;
        auto sw = gameWrapper->GetCurrentGameState();
        if (!sw) return false;
        auto ge = gameWrapper->GetGameEventAsServer();
        return ge && (ge.GetbTraining() || ge.GetbFreeplay());
    }

    void start(){
        if (!inTraining()) { LOG("Start in Freeplay/Training only."); return; }
        active = true; elapsed = 0.0f; M.reset();
        prevYaw = prevPitch = prevSteer = 0.0f; prevYawSign = prevPitchSign = prevSteerSign = 0.0f;
        LOG("Capture started. Use so_stop to end.");
    }

    void status(){
        std::ostringstream oss;
        oss << (active?"ACTIVE":"IDLE") << " | t=" << util::f2(elapsed,1) << "s, samples=" << M.samples;
        LOG(oss.str());
    }

    void stopAndReport(){
        if (!active){ LOG("Not active."); return; }
        active = false;
        computeAndPrintRecommendation();
    }

    void OnInput(std::string /*eventName*/){
        if (!active) return;
        if (!inTraining()) { active=false; LOG("Stopped (left Training/Freeplay)."); return; }

        ServerWrapper sw = gameWrapper->GetGameEventAsServer();
        if (!sw) return;
        CarWrapper car = gameWrapper->GetLocalCar();
        BallWrapper ball = sw.GetBall();
        if (!car || !ball) return;

        float dt = 1.0f / std::max(30.0f, gameWrapper->GetFPS());
        elapsed += dt;

        ControllerInput ci = car.GetInput();
        float yaw = ci.Yaw;
        float pitch = ci.Pitch;
        float steer = ci.Steer;

        Rotator rot = car.GetRotation();
        float yawRad = rot.Yaw * (float)M_PI / 32768.0f;
        Vector fwd = { std::cos(yawRad), std::sin(yawRad), 0.0f };

        Vector carLoc = car.GetLocation();
        Vector ballLoc = ball.GetLocation();
        Vector toBall = { ballLoc.X - carLoc.X, ballLoc.Y - carLoc.Y, 0.0f };
        float dist = util::length2D(toBall);
        Vector dirBall = util::normalize(toBall);

        float errYaw = util::angleDeg2D(fwd, dirBall);
        Vector right = { -fwd.Y, fwd.X, 0.0f };
        float side = util::dot2D(right, dirBall);
        if (side < 0) errYaw = -errYaw;

        static float prevYawRad = yawRad; static bool first=true;
        float angVelYaw = 0.0f;
        if (!first) angVelYaw = (yawRad - prevYawRad) / dt * 180.0f / (float)M_PI;
        first=false; prevYawRad = yawRad;

        M.rot_err_sum += std::abs(errYaw) * dt;
        M.rot_err_max = std::max(M.rot_err_max, std::abs(errYaw));

        float yawSign = (yaw>0) - (yaw<0);
        float errSign = (errYaw>0) - (errYaw<0);
        bool steeringWrongWay = (yawSign != 0 && errSign != 0 && yawSign != errSign);
        float over = steeringWrongWay ? std::abs(yaw) : 0.0f;
        M.oversteer_sum += over * dt;

        float micThresh = 0.12f;
        if (prevYawSign != 0 && yawSign != 0 && yawSign != prevYawSign && std::abs(yaw) < micThresh && std::abs(prevYaw) < micThresh)
            M.microcorr_count += 1.0f;

        static float flickWindow=0.12f;
        static float flickTimer=0.0f; static bool flickArm=false; static float flickPeak=0.0f;
        if (std::abs(yaw) > 0.85f) { flickArm=true; flickTimer=0.0f; flickPeak=0.0f; }
        if (flickArm){
            flickTimer += dt;
            flickPeak = std::max(flickPeak, std::abs(angVelYaw));
            if (flickTimer >= flickWindow){
                M.flick_resp_sum += flickPeak; M.flick_resp_events++;
                flickArm=false; flickTimer=0.0f; flickPeak=0.0f;
            }
        }

        if (car.GetbBallHasBeenHit()) {
            M.touches++;
            Vector bv = ball.GetVelocity();
            Vector cv = car.GetVelocity();
            M.avg_touch_speed += util::length2D({bv.X-cv.X, bv.Y-cv.Y, 0});
        }

        M.samples++;
        prevYaw = yaw; prevYawSign = yawSign; prevSteer = steer; prevSteerSign = (steer>0)-(steer<0);

        if (elapsed >= targetDuration) { active=false; computeAndPrintRecommendation(); }
    }

    struct Recommendation { float steerSensDeltaPct{0}, aerialSensDeltaPct{0}, deadzoneDeltaPct{0}, dodgeDZDeltaPct{0}; };

    void computeAndPrintRecommendation(){
        if (M.samples == 0){ LOG("No samples."); return; }

        float duration = elapsed;
        float oversteer_rate = M.oversteer_sum / std::max(0.001f, duration);
        float micro_rate = M.microcorr_count / std::max(1.0f, duration);
        float rot_err_avg = M.rot_err_sum / std::max(0.001f, duration);
        float flick_avg = (M.flick_resp_events>0) ? (M.flick_resp_sum / M.flick_resp_events) : 0.0f;
        float touch_speed_avg = (M.touches>0) ? (M.avg_touch_speed / M.touches) : 0.0f;

        float n_over = util::clamp(oversteer_rate / 0.12f, 0.f, 2.f);
        float n_micro = util::clamp(micro_rate / 2.0f, 0.f, 2.f);
        float n_rot   = util::clamp(rot_err_avg / 15.0f, 0.f, 2.f);
        float n_flick = util::clamp((300.0f - flick_avg) / 300.0f, 0.f, 2.f);

        Recommendation R;
        float steerAdj = (-0.07f*n_over) + (-0.05f*n_micro) + (-0.05f*n_rot) + ( +0.03f*(1.0f - util::clamp(flick_avg/300.0f,0.f,1.f)) );
        float aerialAdj = (-0.05f*n_over) + (-0.05f*n_micro) + (-0.06f*n_rot) + ( +0.04f*(1.0f - util::clamp(flick_avg/300.0f,0.f,1.f)) );
        float dzAdj = (+0.08f*n_micro) + (+0.05f*n_over) + (+0.05f*(rot_err_avg>12.0f));
        float ddzAdj = (+0.04f*n_over) + (+0.03f*n_micro);

        R.steerSensDeltaPct  = util::clamp(steerAdj, -0.10f, 0.10f) * 100.0f;
        R.aerialSensDeltaPct = util::clamp(aerialAdj, -0.10f, 0.10f) * 100.0f;
        R.deadzoneDeltaPct   = util::clamp(dzAdj,    -0.10f, 0.10f) * 100.0f;
        R.dodgeDZDeltaPct    = util::clamp(ddzAdj,   -0.10f, 0.10f) * 100.0f;

        printReport(n_over, n_micro, rot_err_avg, flick_avg, touch_speed_avg, R);
    }

    void printReport(float n_over, float n_micro, float rot_err_avg, float flick_avg, float touch_speed_avg, const Recommendation& R){
        cvarManager->log("\n=========== RL SensOptimizer Report ===========");
        cvarManager->log("Duration: " + util::f2(elapsed,1) + " s   Samples: " + std::to_string(M.samples));
        cvarManager->log("Avg Rot Error: " + util::f2(rot_err_avg,2) + " deg");
        cvarManager->log("Oversteer Index: " + util::f2(n_over,2) + " (<=1 good)");
        cvarManager->log("Micro-Corrections/s: " + util::f2(n_micro,2));
        cvarManager->log("Flick Peak Avg: " + util::f2(flick_avg,1) + " deg/s (higher is better)");
        cvarManager->log("Avg Touch RelSpeed: " + util::f2(touch_speed_avg,1));
        cvarManager->log("------------------------------------------------");
        cvarManager->log("Recommended adjustments (percent of current):");
        cvarManager->log("  SteeringSensitivity: " + util::f2(R.steerSensDeltaPct,1) + "%");
        cvarManager->log("  AerialSensitivity:   " + util::f2(R.aerialSensDeltaPct,1) + "%");
        cvarManager->log("  Deadzone:            " + util::f2(R.deadzoneDeltaPct,1) + "%");
        cvarManager->log("  Dodge Deadzone:      " + util::f2(R.dodgeDZDeltaPct,1) + "%");
        cvarManager->log("================================================\n");
    }
};

BAKKESMOD_PLUGIN(RLSensOptimizer, "RL SensOptimizer – measures & recommends sensitivity/deadzones", "1.0.0", PLUGINTYPE_FREEPLAY)
