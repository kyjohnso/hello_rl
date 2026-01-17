use bevy::{input::mouse::{MouseMotion, MouseWheel}, prelude::*};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Hello RL - Bevy".to_string(),
                resolution: (800, 600).into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(Update, close_on_esc)
        .add_systems(Update, orbit_camera)
        .run();
}

#[derive(Component)]
struct OrbitCamera {
    pub focus: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

// System to close window on ESC key
fn close_on_esc(
    mut commands: Commands,
    focused_windows: Query<(Entity, &Window)>,
    input: Res<ButtonInput<KeyCode>>,
) {
    for (window, _) in focused_windows.iter() {
        if input.just_pressed(KeyCode::Escape) {
            commands.entity(window).despawn();
        }
    }
}

// Orbit camera system
fn orbit_camera(
    mut query: Query<(&mut OrbitCamera, &mut Transform)>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut mouse_wheel: MessageReader<MouseWheel>,
) {
    let Ok((mut orbit, mut transform)) = query.single_mut() else {
        return;
    };

    // Handle mouse drag for rotation
    if mouse_button.pressed(MouseButton::Left) {
        for motion in mouse_motion.read() {
            orbit.yaw -= motion.delta.x * 0.003;
            orbit.pitch -= motion.delta.y * 0.003;
            orbit.pitch = orbit.pitch.clamp(-1.5, 1.5);
        }
    } else {
        // Clear the motion events if not rotating
        mouse_motion.clear();
    }

    // Handle mouse wheel for zoom
    for wheel in mouse_wheel.read() {
        orbit.distance -= wheel.y * 0.5;
        orbit.distance = orbit.distance.clamp(2.0, 20.0);
    }

    // Calculate new camera position
    let rot = Quat::from_euler(EulerRot::YXZ, orbit.yaw, orbit.pitch, 0.0);
    let offset = rot * Vec3::new(0.0, 0.0, orbit.distance);
    transform.translation = orbit.focus + offset;
    transform.look_at(orbit.focus, Vec3::Y);
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Spawn an orbit camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitCamera {
            focus: Vec3::ZERO,
            distance: 5.0,
            yaw: 0.0,
            pitch: 0.4,
        },
    ));

    // Spawn a light
    commands.spawn((
        PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // Spawn a ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(10.0, 10.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
    ));

    // Spawn a cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));
}
