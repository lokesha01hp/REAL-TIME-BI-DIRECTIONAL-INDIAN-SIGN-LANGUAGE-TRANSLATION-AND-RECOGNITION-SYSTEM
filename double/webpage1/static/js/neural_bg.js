const canvas = document.getElementById('bgCanvas');
if (canvas) {
    const ctx = canvas.getContext('2d');
    canvas.width = innerWidth;
    canvas.height = innerHeight;
    const nodes = Array.from({length: 100}, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.7,
        vy: (Math.random() - 0.5) * 0.7
    }));
    function animate() {
        ctx.fillStyle = 'rgba(0,0,0,0.05)';
        ctx.fillRect(0,0,canvas.width,canvas.height);
        nodes.forEach(n => {
            n.x += n.vx; n.y += n.vy;
            if (n.x < 0 || n.x > canvas.width) n.vx *= -1;
            if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
            ctx.beginPath();
            ctx.arc(n.x, n.y, 3, 0, Math.PI*2);
            ctx.fillStyle = '#bb86fc';
            ctx.fill();
        });
        requestAnimationFrame(animate);
    }
    animate();
    window.onresize = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
}