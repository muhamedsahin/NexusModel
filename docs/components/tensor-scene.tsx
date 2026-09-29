"use client";
import { useEffect, useRef, useState } from "react";
import type { Locale } from "@/lib/types";
export function TensorScene({ lang }: { lang: Locale }) {
  const host = useRef<HTMLDivElement>(null);
  const [fallback, setFallback] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const expandedRef = useRef(false);
  function toggleExpand() {
    expandedRef.current = !expandedRef.current;
    setExpanded(expandedRef.current);
    window.dispatchEvent(new Event("nx-motion"));
  }
  useEffect(() => {
    let dispose = () => {};
    let cancelled = false;
    import("three")
      .then((THREE) => {
        if (cancelled || !host.current) return;
        const el = host.current;
        let renderer: InstanceType<typeof THREE.WebGLRenderer>;
        try {
          renderer = new THREE.WebGLRenderer({
            alpha: true,
            antialias: true,
            powerPreference: "low-power",
          });
        } catch {
          setFallback(true);
          return;
        }
        renderer.setPixelRatio(Math.min(devicePixelRatio, 1.7));
        renderer.setClearColor(0, 0);
        el.appendChild(renderer.domElement);
        renderer.domElement.setAttribute("aria-hidden", "true");
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 60);
        camera.position.set(0, 0.25, 12);
        const group = new THREE.Group();
        scene.add(group);
        const cubeGeo = new THREE.BoxGeometry(0.42, 0.42, 0.42);
        const edgeGeo = new THREE.EdgesGeometry(cubeGeo);
        const edgeMat = new THREE.LineBasicMaterial({
          color: 0x9de95f,
          transparent: true,
          opacity: 0.7,
        });
        const cubeMat = new THREE.MeshPhysicalMaterial({
          color: 0x658c40,
          metalness: 0.7,
          roughness: 0.24,
          transparent: true,
          opacity: 0.36,
        });
        const coreMat = new THREE.MeshStandardMaterial({
          color: 0xb9fc75,
          emissive: 0x64af2b,
          emissiveIntensity: 0.35,
          metalness: 0.4,
          roughness: 0.3,
        });
        const cubes: InstanceType<typeof THREE.Group>[] = [];
        const origins: InstanceType<typeof THREE.Vector3>[] = [];
        for (let x = -2; x <= 2; x++)
          for (let y = -2; y <= 2; y++)
            for (let z = -2; z <= 2; z++) {
              const c = new THREE.Group();
              c.position.set(x * 0.68, y * 0.68, z * 0.68);
              const core = x === 0 && y === 0 && z === 0;
              c.add(new THREE.Mesh(cubeGeo, core ? coreMat : cubeMat));
              c.add(new THREE.LineSegments(edgeGeo, edgeMat));
              group.add(c);
              cubes.push(c);
              origins.push(c.position.clone());
            }
        const positions: number[] = [];
        for (let axis = 0; axis < 3; axis++)
          for (let a = -2; a <= 2; a++)
            for (let b = -2; b <= 2; b++) {
              const from = [a * 0.68, b * 0.68, -1.75],
                to = [a * 0.68, b * 0.68, 1.75];
              if (axis === 0) {
                from.reverse();
                to.reverse();
              }
              if (axis === 1) {
                [from[1], from[2]] = [from[2], from[1]];
                [to[1], to[2]] = [to[2], to[1]];
              }
              positions.push(...from, ...to);
            }
        const lineGeo = new THREE.BufferGeometry();
        lineGeo.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(positions, 3),
        );
        const lineMat = new THREE.LineBasicMaterial({
          color: 0x8bc45c,
          transparent: true,
          opacity: 0.14,
        });
        group.add(new THREE.LineSegments(lineGeo, lineMat));
        const orbitGeo = new THREE.BufferGeometry();
        const orbitPoints: number[] = [];
        for (let i = 0; i < 180; i++) {
          const a = (i / 180) * Math.PI * 2;
          orbitPoints.push(Math.cos(a) * 3.1, 0, Math.sin(a) * 3.1);
        }
        orbitGeo.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(orbitPoints, 3),
        );
        const orbitMat = new THREE.LineBasicMaterial({
          color: 0x829d70,
          transparent: true,
          opacity: 0.25,
        });
        const orbit = new THREE.LineLoop(orbitGeo, orbitMat);
        orbit.rotation.x = 0.25;
        scene.add(orbit);
        const dotsGeo = new THREE.BufferGeometry();
        const dots = new Float32Array(45 * 3);
        for (let i = 0; i < 45; i++) {
          dots[i * 3] = Math.sin(i * 32.2) * 3.2;
          dots[i * 3 + 1] = Math.cos(i * 17.9) * 2.6;
          dots[i * 3 + 2] = Math.sin(i * 9.1) * 2;
        }
        dotsGeo.setAttribute("position", new THREE.BufferAttribute(dots, 3));
        const dotsMat = new THREE.PointsMaterial({
          color: 0xc4fc91,
          size: 0.035,
          transparent: true,
          opacity: 0.8,
        });
        const particles = new THREE.Points(dotsGeo, dotsMat);
        scene.add(particles);
        scene.add(new THREE.AmbientLight(0xffffff, 2));
        const light = new THREE.DirectionalLight(0xd3ffbc, 5);
        light.position.set(4, 6, 4);
        scene.add(light);
        const back = new THREE.DirectionalLight(0x7d9bff, 2);
        back.position.set(-4, -2, -4);
        scene.add(back);
        const pulseGeo = new THREE.SphereGeometry(0.045, 8, 8);
        const pulseMat = new THREE.MeshBasicMaterial({ color: 0xe1ffbe });
        const pulses = Array.from({ length: 12 }, () => {
          const p = new THREE.Mesh(pulseGeo, pulseMat);
          group.add(p);
          return p;
        });
        let frame = 0;
        let visible = true;
        let paused = document.documentElement.dataset.paused === "true";
        let elapsed = 0;
        let last = 0;
        let pointer = { x: 0, y: 0 };
        const draw = (now = 0) => {
          frame = 0;
          if (cancelled) return;
          const dt = Math.min((now - last) / 1000, 0.05);
          last = now;
          if (!paused) elapsed += dt;
          group.rotation.set(
            -0.3 + pointer.y * 0.12,
            0.6 + elapsed * 0.08 + pointer.x * 0.18,
            0.12,
          );
          group.position.y = Math.sin(elapsed * 0.6) * 0.08;
          particles.rotation.y = elapsed * 0.04;
          const wave = 1 + Math.sin(elapsed * 0.6) * 0.035;
          group.scale.setScalar(wave);
          const target = expandedRef.current ? 1.48 : 1;
          for (let i = 0; i < cubes.length; i++) {
            const desired = origins[i];
            if (paused) cubes[i].position.copy(desired).multiplyScalar(target);
            else {
              cubes[i].position.x +=
                (desired.x * target - cubes[i].position.x) * 0.06;
              cubes[i].position.y +=
                (desired.y * target - cubes[i].position.y) * 0.06;
              cubes[i].position.z +=
                (desired.z * target - cubes[i].position.z) * 0.06;
            }
          }
          pulses.forEach((p, i) => {
            const phase = ((elapsed * 0.6 + i * 0.31) % 3.4) - 1.7;
            p.position.set(
              phase,
              ((i % 5) - 2) * 0.68,
              ((Math.floor(i / 5) % 5) - 2) * 0.68,
            );
          });
          renderer.render(scene, camera);
          if (visible && !document.hidden && !paused)
            frame = requestAnimationFrame(draw);
        };
        const wake = () => {
          if (frame) cancelAnimationFrame(frame);
          frame = 0;
          last = performance.now();
          draw(last);
        };
        const resize = new ResizeObserver(() => {
          const w = el.clientWidth,
            h = el.clientHeight;
          renderer.setSize(w, h);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          wake();
        });
        resize.observe(el);
        const io = new IntersectionObserver(([e]) => {
          visible = e.isIntersecting;
          if (visible) wake();
          else if (frame) {
            cancelAnimationFrame(frame);
            frame = 0;
          }
        });
        io.observe(el);
        const move = (e: PointerEvent) => {
          if (paused) return;
          const r = el.getBoundingClientRect();
          pointer = {
            x: (e.clientX - r.left) / r.width - 0.5,
            y: (e.clientY - r.top) / r.height - 0.5,
          };
        };
        const motion = () => {
          paused = document.documentElement.dataset.paused === "true";
          wake();
        };
        const theme = () => {
          const light = document.documentElement.dataset.theme === "light";
          edgeMat.color.set(light ? 0x426b2a : 0x9de95f);
          cubeMat.color.set(light ? 0x648b42 : 0x658c40);
          lineMat.opacity = light ? 0.28 : 0.14;
          wake();
        };
        const visibility = () => {
          if (document.hidden) {
            if (frame) cancelAnimationFrame(frame);
            frame = 0;
          } else wake();
        };
        el.addEventListener("pointermove", move);
        window.addEventListener("nx-motion", motion);
        window.addEventListener("nx-theme", theme);
        document.addEventListener("visibilitychange", visibility);
        theme();
        dispose = () => {
          cancelAnimationFrame(frame);
          resize.disconnect();
          io.disconnect();
          el.removeEventListener("pointermove", move);
          window.removeEventListener("nx-motion", motion);
          window.removeEventListener("nx-theme", theme);
          document.removeEventListener("visibilitychange", visibility);
          [cubeGeo, edgeGeo, lineGeo, orbitGeo, dotsGeo, pulseGeo].forEach(
            (x) => x.dispose(),
          );
          [
            edgeMat,
            cubeMat,
            coreMat,
            lineMat,
            orbitMat,
            dotsMat,
            pulseMat,
          ].forEach((x) => x.dispose());
          renderer.dispose();
          renderer.domElement.remove();
        };
      })
      .catch(() => setFallback(true));
    return () => {
      cancelled = true;
      dispose();
    };
  }, []);
  return (
    <div className="tensor-scene">
      <div
        ref={host}
        className="webgl-host"
        role="img"
        aria-label={
          lang === "tr"
            ? "Dönen üç boyutlu tensör kafesi; kavramsal görselleştirme"
            : "Rotating three-dimensional tensor lattice; conceptual visualization"
        }
      />
      {fallback && (
        <div className="tensor-fallback">
          <span>[ 5 × 5 × 5 ]</span>
          <div>
            {Array.from({ length: 25 }, (_, i) => (
              <i key={i} />
            ))}
          </div>
        </div>
      )}
      <div className="scene-label label-top">
        <span className="tiny-cross">+</span> TENSOR&lt;float32&gt;
        <small>SHAPE [5, 5, 5]</small>
      </div>
      <div className="scene-label label-bottom">
        <span className="signal-dot" />
        64-BYTE ALIGNED
        <small>
          {lang === "tr" ? "PAYLAŞIMLI DEPOLAMA" : "SHARED STORAGE"}
        </small>
      </div>
      <div className="scene-coordinate">
        X 01.00
        <br />Y 00.64
        <br />Z 00.32
      </div>
      <button
        className="scene-control"
        onClick={toggleExpand}
        aria-pressed={expanded}
      >
        <span>{expanded ? "−" : "+"}</span>
        {lang === "tr"
          ? expanded
            ? "Birleştir"
            : "Katmanları ayır"
          : expanded
            ? "Collapse layers"
            : "Expand layers"}
      </button>
      <div className="scene-caption">
        {lang === "tr"
          ? "01 / KAVRAMSAL TENSÖR GÖRÜNÜMÜ"
          : "01 / CONCEPTUAL TENSOR VIEW"}
      </div>
    </div>
  );
}
