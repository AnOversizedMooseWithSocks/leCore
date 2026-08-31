"""Part 09b of UnifiedMind's faculty surface -- the scene verbs (sky_model .. refine_scene), split out in sweep 114 when the parent part crossed the
2,000-line budget test_unified_split pins (the point of the split was file size; a
part that grows past the cap gets split again, never a raised cap).

NOT A STANDALONE MODULE. One slice of the single `UnifiedMind` class, assembled by
holographic/misc/holographic_unified.py, which is still the only import path anyone
uses. Carries no `__init__`; assumes the state UnifiedMind.__init__ sets up.
"""
import numpy as np

from holographic.unified import check_part


class _UnifiedPart09B:

    def sky_model(self, hour=12.0, clouds=(), stars_seed=None, star_density=0.9985, moon=None,
                  sun_intensity=18.0, cloud_seed=0, time_s=0.0, wind=(0.05, 0.02), evolve=0.10):
        """A PARAMETRIC sky -- time of day, sun arc, moon, deterministic stars, HIGH cloud layers -- as one
        radiance callable f(dirs)->rgb, pluggable anywhere sky_dome is (the tracer's sky=, a dome light's
        color=). hour drives a keyed gradient palette AND the sun's position; clouds=[(kind, coverage)] with SEVEN kinds -- cirrus
        (streaks), cirrostratus (veil), cirrocumulus (mackerel sky), altocumulus (clumps), altostratus
        (milky sun), stratocumulus (broken deck), nimbostratus (rain blanket) -- composing as Beer-Lambert
        shells with per-KIND extinction and threshold sharpness (cellular kinds keep real GAPS, base
        shading keeps texture in even a full deck);
        stars_seed makes a hash-of-direction starfield (same seed = same sky forever) that fades by daylight
        and by cloud; moon=True auto-places opposite the sun. MEASURED under one fixed transform: noon
        linear mean 0.529, sunset+cirrus 0.260, midnight+stars 0.019 -- a 28x day/night range the
        auto-exposing display view will happily hide, so compare skies with view=None. LOW puffy clouds are
        deliberately refused toward cloud_scene (the volumetric stack -- real depth and self-shadowing);
        this model owns only what reads as a textured dome from the ground. See holographic_skymodel."""
        from holographic.rendering.holographic_skymodel import sky_model
        return sky_model(hour=hour, clouds=clouds, stars_seed=stars_seed, star_density=star_density,
                         moon=moon, sun_intensity=sun_intensity, cloud_seed=cloud_seed,
                         time_s=time_s, wind=tuple(wind), evolve=evolve)

    def render_animation(self, scene, camera, keys, n_frames=24, fps=12.0, width=160, height=120,
                         gif=None, interp="smooth", seed=0, lights=None, sky=None, sky_keys=None,
                         **render_kw):
        """ANIMATE the Scene document and render it -- keyframes in, frames (and optionally a GIF) out.

        NOTHING HERE IS NEW MACHINERY, and that is the point. The keyframe Timeline (key/sample, easing,
        vectorised) existed. place() existed. render_preview existed. save_gif was the one missing writer.
        What did not exist was any path that COMPOSES them -- 'animate an object in my scene document'
        returned scene_info, 'render frames over time' returned a cache -- and no JSON client could build
        the composition itself, because a Timeline object cannot cross POST /invoke. This is the bridge, in
        the same sense scene_set_texture is: JSON-safe values in, the callables live server-side.

        `keys` (JSON-safe): {handle: {property: [[t, value], ...]}} with property one of
        'position' ([x,y,z]), 'rotation' (Euler degrees [rx,ry,rz]), 'scale' (a number). Times are in
        SECONDS; the animation spans [0, n_frames/fps]. `interp` is the Timeline's easing for every key
        ('linear', 'step', 'smooth', 'ease_in', 'ease_out').

        Returns the list of rendered frames ((H,W,3) float each); with `gif=` set, also writes an animated
        GIF there -- the see->fix loop for MOTION.

        KEPT NEGATIVES, stated rather than discovered later:
          * Frames render at PREVIEW quality (draft, 1 bounce) -- measured at 12x the full path. An
            N-frame final-quality animation is N full renders and wants the job system, not a loop that
            holds an HTTP request open for an hour.
          * Edits go through place(), so the LAST FRAME'S transforms persist on the document afterwards --
            deliberate (undo works, and 'where did it end up' is a real question), but a caller re-rendering
            stills afterwards should place() things back or undo.
          * Rotation keys interpolate EULER ANGLES componentwise. Fine for turntables and tilts; a path
            that swings past gimbal territory wants quaternions, which the Timeline does not speak. That is
            a real limitation of composing existing parts and it is documented instead of hidden."""
        import numpy as np
        from holographic.rendering.holographic_scene_render import render_preview

        tl = self.timeline()
        tracks = []                                            # (handle, property, channel_name)
        for handle, props in keys.items():
            for prop, kvs in props.items():
                if prop not in ("position", "rotation", "scale"):
                    raise ValueError("unknown animatable property %r -- position, rotation, scale "
                                     "(what place() can apply)" % (prop,))
                chan = "%s.%s" % (handle, prop)
                for t, value in kvs:
                    tl.key(chan, float(t), np.asarray(value, float) if prop != "scale" else float(value),
                           interp=interp)
                tracks.append((handle, prop, chan))

        # sky_keys: the TIMELAPSE half, default off (additive). {'hour': [[t, hour], ...]} plus any
        # static sky_model kwargs ('clouds', 'stars_seed', 'moon', ...). Discovery already routed
        # 'day to night timelapse' at sky_model + render_animation -- but neither could DO it: keys move
        # objects, and the sky was frozen per call. The hour rides the SAME Timeline as the object keys
        # (delegation, per the hand-roll audit), and the sky closure is rebuilt per frame -- closure
        # construction only, the texture fields inside are built per call by sky_model as before.
        sky_tl = None
        sky_static = {}
        if sky_keys is not None:
            if sky is not None:
                raise ValueError("pass sky= (fixed) OR sky_keys= (animated), not both -- one sky per frame")
            sky_static = {k: v for k, v in sky_keys.items() if k != "hour"}
            if "hour" not in sky_keys:
                raise ValueError("sky_keys needs 'hour': [[t, hour], ...] -- the keyed part; everything "
                                 "else in sky_keys is passed to sky_model unchanged")
            sky_tl = self.timeline()
            for tt, hh in sky_keys["hour"]:
                sky_tl.key("hour", float(tt), float(hh), interp=interp)

        frames = []
        for i in range(int(n_frames)):
            t = i / float(fps)
            for handle, prop, chan in tracks:
                self.place(scene, handle, **{prop: tl.sample(chan, t)})
            frame_sky = sky
            frame_lights = lights
            if sky_tl is not None:
                # clouds MOVE during a timelapse (review: "changing shape and moving naturally"): the
                # frame time feeds sky_model's wind-drift + solid-noise evolution unless the caller pinned
                # time_s themselves. A timelapse compresses hours into seconds, so the animation time is
                # scaled up (x240: one real second of animation ~ four minutes of sky) -- override with an
                # explicit 'time_scale' in sky_keys, or freeze the clouds with 'time_s': 0.
                if "time_s" not in sky_static:
                    sky_static_frame = dict(sky_static)
                    scale_t = float(sky_static_frame.pop("time_scale", 240.0))
                    sky_static_frame["time_s"] = t * scale_t
                else:
                    sky_static_frame = {k: v for k, v in sky_static.items() if k != "time_scale"}
                frame_sky = self.sky_model(hour=float(sky_tl.sample("hour", t)), **sky_static_frame)
                # a timelapse whose LIGHTING ignores the sky is two different times of day in one frame;
                # if the caller gave no lights, the animated sky drives a dome so the ground follows the sky
                if lights is None:
                    frame_lights = [self.scene_light("dome", color=frame_sky, intensity=1.0)]
            frames.append(np.asarray(render_preview(scene, camera, width=width, height=height,
                                                    seed=seed, lights=frame_lights, sky=frame_sky,
                                                    **render_kw), float))
        if gif is not None:
            from holographic.rendering.holographic_render import save_gif
            save_gif(gif, frames, fps=fps)
        return frames

    def describe_to_scene(self, text, scene=None):
        """Words -> the CANONICAL Scene document: 'a red cube on the left and a green sphere on the right'
        becomes real, named, handled objects you can then texture, place, keyframe, and path-trace.

        WHY A BRIDGE, when build_scene already exists. leCore grew TWO scene systems that could not talk:
        build_scene -> a SemanticScene (parses text, resolves 'on'/'beside'/'inside' into positions, renders
        itself, adjusts by sentence) and new_scene -> the Scene DOCUMENT (handles, undo, selection -- the
        thing scene_set_texture, place, render_animation, render_scene_document and the whole HTTP surface
        operate on). Every parity capability of this arc landed on the document side, so an agent that
        started from words was cut off from all of it: 8/8 audit phrasings for this conversion returned
        nothing relevant. The parser and the layout heuristics are REUSED (interpret_description +
        realize_scene), not reimplemented -- this is a join, and both halves keep their own behaviour.

        Returns {scene, handles: {object name: handle}, unknown: [words the parser could not ground],
        suggestions: [...]} -- unknown words are REPORTED rather than silently dropped, because 'a wooden
        gnome' quietly becoming an empty scene is the kind of no-op that sends an agent debugging its
        camera. Pass `scene=` to add into an existing document instead of a fresh one.

        Parsed colours become per-object PBRMaterials (base colour + modest roughness), so 'red' survives
        into the path tracer without the caller mapping words to library names.

        KEPT NEGATIVES, inherited honestly from the halves rather than papered over:
          * realize_scene's own limitation stands: rotation is not modelled -- 'diagonal' is an offset +
            stretch, not a tilt. Fix it there if it matters; a bridge is the wrong layer.
          * The realized SDFs arrive PRE-PLACED (position baked into the geometry), so each object's
            document transform starts as identity. place()/keyframes still work -- they COMPOSE on top --
            but scene_info reports position [0,0,0] for a freshly described object. Re-deriving the baked
            offset to normalise it would mean probing the SDF for its own centre: guessy, and wrong for
            'inside'. Reported as-is instead.
          * The controlled vocabulary is the parser's (SHAPES/COLORS/RELATIONS in scene_semantic). This
            bridge adds no words; `unknown` tells you what fell through."""
        from holographic.simulation_and_physics.holographic_scene_semantic import (interpret_description,
                                                                                   realize_scene)
        import holographic.materials_and_texture.holographic_matlib as ML

        parsed = interpret_description(text)
        renderables = realize_scene(parsed["objects"]) if parsed["objects"] else []
        sc = scene if scene is not None else self.new_scene()
        handles = {}
        for r in renderables:
            col = tuple(r.get("color", (0.7, 0.7, 0.7)))
            mat = r.get("mat_name") or ML.PBRMaterial(name="described:%s" % r["name"],
                                                      base_color=col + (1.0,), roughness=0.6)
            handles[r["name"]] = sc.add(name=r["name"], geometry=r["sdf"], material=mat)
        return {"scene": sc, "handles": handles,
                "unknown": list(parsed.get("unknown", ())),
                "suggestions": list(parsed.get("suggestions", ()))}

    def refine_scene(self, scene, target_image, max_steps=4, apply=True, geometry=False, focus=None,
                     width=96, height=72):
        """CLOSE THE LOOP: hand a described scene a TARGET IMAGE and let the engine improve itself toward
        it -- the capability that goes past screenshot-and-hope. The reference Blender integration can show
        an agent its render; it cannot score candidate edits against a goal and apply the best one. This
        can, deterministically, with the trail on record.

        `scene` is a SemanticScene (from build_scene / describe-side work); `target_image` is (H,W,3) --
        MUST match `width` x `height`, because the critic compares like with like (a size mismatch used to
        surface as a broadcast error from deep inside SSIM; it is checked HERE now, with the fix named).
        apply=True runs the bounded greedy driver (refine_to_target) and returns {applied,
        start_distance, final_distance, steps, history}; apply=False only SCORES (propose_edits) and
        returns the ranked candidates without touching the scene -- the read-only critic an agent can
        consult before deciding. geometry=True lets it also move/scale objects; `focus` scores a subject
        region only.

        Verified live before wiring: from 'a red sphere' toward a night-time target, distance 0.2625 ->
        0.0000 in one applied edit -- the loop rediscovered 'make it night' by itself.

        KEPT NEGATIVES: the edit vocabulary is the semantic scene's (lighting/brightness/material/colour,
        coarse move/scale with geometry=True) -- it proposes from what it can say, not from arbitrary
        parameter space; and it operates on the SEMANTIC scene, not the Scene document, because candidate
        edits are sentences. Use describe_to_scene afterwards to promote the refined result. See
        holographic_scene_semantic.SemanticScene.refine_to_target / propose_edits."""
        import numpy as np
        tgt = np.asarray(target_image, float)
        if tgt.shape[:2] != (height, width):
            raise ValueError("target_image is %dx%d but the critic renders %dx%d -- pass a target at the "
                             "loop's own size (width=/height=), or set width/height to match the target"
                             % (tgt.shape[1], tgt.shape[0], width, height))
        if apply:
            return scene.refine_to_target(tgt, max_steps=max_steps, geometry=geometry, focus=focus,
                                          width=width, height=height)
        return scene.propose_edits(tgt, geometry=geometry, width=width, height=height)



def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p09b_scene_verbs", "_UnifiedPart09B")
    print("holographic_unified_p09b_scene_verbs selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()
